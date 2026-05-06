package executor

import (
	"bufio"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/md5"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	_ "embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	qoderProvider              = "qoder"
	qoderDefaultModel          = "lite"
	qoderVersion               = "0.1.43"
	qoderAppCode               = "cosy"
	qoderSecret                = "d2FyLCB3YXIgbmV2ZXIgY2hhbmdlcw=="
	qoderClientType            = "5"
	qoderClientIP              = "169.254.198.161"
	qoderUserAgent             = "Go-http-client/2.0"
	qoderDefaultModelSource    = "system"
	qoderControlBaseURL        = "https://center.qoder.sh"
	qoderChatURL               = "https://api3.qoder.sh/algo/api/v2/service/pro/sse/agent_chat_generation?FetchKeys=llm_model_result&AgentId=agent_common&Encode=1"
	qoderJobTokenPath          = "/algo/api/v3/user/jobToken?Encode=1"
	qoderHeartbeatPath         = "/algo/api/v1/heartbeat?Encode=1"
	qoderStreamIdleTimeout     = 3 * time.Second
	qoderRequestTimeout        = 30 * time.Second
	qoderControlRequestTimeout = 15 * time.Second
)

const qoderServerPublicKeyPEM = `-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDA8iMH5c02LilrsERw9t6Pv5Nc
4k6Pz1EaDicBMpdpxKduSZu5OANqUq8er4GM95omAGIOPOh+Nx0spthYA2BqGz+l
6HRkPJ7S236FZz73In/KVuLnwI8JJ2CbuJap8kvheCCZpmAWpb/cPx/3Vr/J6I17
XcW+ML9FoCI6AOvOzwIDAQAB
-----END PUBLIC KEY-----`

//go:embed qoder_baseprompt.json
var qoderBasePromptTemplate []byte

// QoderExecutor implements Qoder's native signing, encoding and streaming protocol.
// The auth api_key attribute is treated as the Qoder personal access token.
type QoderExecutor struct {
	cfg *config.Config

	mu       sync.Mutex
	sessions map[string]*qoderSession
}

type qoderSession struct {
	identity     qoderAuthIdentity
	machineID    string
	machineToken string
	machineType  string
	cosyKey      string
	info         string
	createdAt    time.Time
}

type qoderAuthIdentity struct {
	Name               string `json:"name"`
	AID                string `json:"aid"`
	UID                string `json:"uid"`
	YxUID              string `json:"yx_uid"`
	OrganizationID     string `json:"organization_id"`
	OrganizationName   string `json:"organization_name"`
	UserType           string `json:"user_type"`
	SecurityOauthToken string `json:"security_oauth_token"`
	RefreshToken       string `json:"refresh_token"`
}

// NewQoderExecutor creates a native Qoder executor instead of routing through OpenAI-compatible mode.
func NewQoderExecutor(cfg *config.Config) *QoderExecutor {
	return &QoderExecutor{cfg: cfg, sessions: make(map[string]*qoderSession)}
}

// Identifier implements cliproxyauth.ProviderExecutor.
func (e *QoderExecutor) Identifier() string { return qoderProvider }

// Execute converts the caller payload to a Qoder prompt, collects Qoder SSE deltas,
// and returns a non-streaming response matching the caller's original protocol.
func (e *QoderExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	if opts.Alt == "responses/compact" {
		return resp, statusErr{code: http.StatusNotImplemented, msg: "/responses/compact not supported by qoder"}
	}
	baseModel := qoderNormalizeModel(req.Model)
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	payload := qoderOriginalPayload(req, opts)
	prompt := qoderPromptFromPayload(payload, opts)
	traceID := qoderShortID()
	log.Infof("qoder request start trace=%s stream=false model=%s promptChars=%d", traceID, baseModel, len(prompt))

	var text strings.Builder
	begin := time.Now()
	if err = e.streamQoderText(ctx, auth, baseModel, prompt, traceID, func(delta string) {
		text.WriteString(delta)
	}); err != nil {
		return resp, err
	}
	output := text.String()
	log.Infof("qoder request done trace=%s stream=false model=%s outputChars=%d costMs=%d", traceID, baseModel, len(output), time.Since(begin).Milliseconds())
	reporter.EnsurePublished(ctx)
	return cliproxyexecutor.Response{Payload: qoderNonStreamResponse(opts.SourceFormat.String(), baseModel, output)}, nil
}

// ExecuteStream converts Qoder deltas into the streaming envelope expected by
// OpenAI chat completions, OpenAI Responses, or Anthropic Messages handlers.
func (e *QoderExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := qoderNormalizeModel(req.Model)
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	payload := qoderOriginalPayload(req, opts)
	prompt := qoderPromptFromPayload(payload, opts)
	traceID := qoderShortID()
	log.Infof("qoder request start trace=%s stream=true format=%s model=%s promptChars=%d", traceID, opts.SourceFormat.String(), baseModel, len(prompt))

	httpResp, err := e.openQoderStream(ctx, auth, baseModel, prompt, traceID)
	if err != nil {
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go e.forwardQoderStream(ctx, httpResp, out, opts.SourceFormat.String(), baseModel, traceID)
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

// Refresh keeps the configured Qoder PAT unchanged. A new bearer session is
// established lazily on the next request if the cached session was cleared.
func (e *QoderExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	if auth == nil {
		return nil, fmt.Errorf("qoder executor: auth is nil")
	}
	_, err := e.getOrCreateSession(ctx, auth)
	if err != nil {
		return nil, err
	}
	return auth, nil
}

// CountTokens mirrors qoder2api behavior for Claude Code token-count probes.
func (e *QoderExecutor) CountTokens(context.Context, *cliproxyauth.Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{Payload: []byte(`{"input_tokens":0}`)}, nil
}

// HttpRequest is intentionally unsupported because Qoder requests must be signed
// with the encoded body and API path together, which generic passthrough calls cannot do safely.
func (e *QoderExecutor) HttpRequest(context.Context, *cliproxyauth.Auth, *http.Request) (*http.Response, error) {
	return nil, statusErr{code: http.StatusNotImplemented, msg: "qoder executor does not support generic HTTP passthrough"}
}

// streamQoderText opens the native Qoder SSE endpoint and emits extracted text deltas.
func (e *QoderExecutor) streamQoderText(ctx context.Context, auth *cliproxyauth.Auth, model, prompt, traceID string, onText func(string)) error {
	httpResp, err := e.openQoderStream(ctx, auth, model, prompt, traceID)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := httpResp.Body.Close(); closeErr != nil {
			log.Debugf("qoder close stream body trace=%s error=%v", traceID, closeErr)
		}
	}()
	return consumeQoderTextStream(ctx, httpResp.Body, traceID, onText)
}

// openQoderStream builds the signed native Qoder request using the current bearer session.
func (e *QoderExecutor) openQoderStream(ctx context.Context, auth *cliproxyauth.Auth, model, prompt, traceID string) (*http.Response, error) {
	sess, err := e.getOrCreateSession(ctx, auth)
	if err != nil {
		return nil, err
	}
	body, modelSource, err := buildQoderBody(model, prompt, sess.identity.UserType)
	if err != nil {
		return nil, err
	}
	encodedBody, err := qoderEncode(body)
	if err != nil {
		return nil, err
	}
	bearer, cosyDate, err := buildQoderBearer(sess, encodedBody, "/api/v2/service/pro/sse/agent_chat_generation")
	if err != nil {
		return nil, err
	}
	reqCtx := ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, qoderChatURL, strings.NewReader(encodedBody))
	if err != nil {
		return nil, err
	}
	applyQoderStreamHeaders(httpReq, sess, bearer, cosyDate, model, modelSource)

	authID, authLabel, authType, authValue := "", "", "", ""
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       qoderChatURL,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      []byte(encodedBody),
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
	log.Debugf("qoder upstream start trace=%s model=%s encodedBodyChars=%d", traceID, model, len(encodedBody))

	httpClient := helps.NewProxyAwareHTTPClient(reqCtx, e.cfg, auth, qoderRequestTimeout)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		defer httpResp.Body.Close()
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		return nil, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}
	return httpResp, nil
}

// forwardQoderStream wraps native Qoder deltas into the downstream stream format.
func (e *QoderExecutor) forwardQoderStream(ctx context.Context, httpResp *http.Response, out chan<- cliproxyexecutor.StreamChunk, sourceFormat, model, traceID string) {
	defer close(out)
	defer func() {
		if httpResp != nil && httpResp.Body != nil {
			_ = httpResp.Body.Close()
		}
	}()
	if httpResp == nil || httpResp.Body == nil {
		out <- cliproxyexecutor.StreamChunk{Err: statusErr{code: http.StatusBadGateway, msg: "qoder executor: missing stream body"}}
		return
	}
	begin := time.Now()
	format := strings.ToLower(strings.TrimSpace(sourceFormat))
	emitter := newQoderStreamEmitter(format, model)
	for _, payload := range emitter.start() {
		out <- cliproxyexecutor.StreamChunk{Payload: payload}
	}
	chunks := 0
	chars := 0
	err := consumeQoderTextStream(ctx, httpResp.Body, traceID, func(delta string) {
		chunks++
		chars += len(delta)
		for _, payload := range emitter.delta(delta) {
			out <- cliproxyexecutor.StreamChunk{Payload: payload}
		}
	})
	if err != nil {
		out <- cliproxyexecutor.StreamChunk{Err: err}
		return
	}
	for _, payload := range emitter.done() {
		out <- cliproxyexecutor.StreamChunk{Payload: payload}
	}
	log.Infof("qoder request done trace=%s stream=true format=%s model=%s chunks=%d outputChars=%d costMs=%d", traceID, format, model, chunks, chars, time.Since(begin).Milliseconds())
}

// getOrCreateSession exchanges the Qoder PAT for job tokens and caches the native bearer session.
func (e *QoderExecutor) getOrCreateSession(ctx context.Context, auth *cliproxyauth.Auth) (*qoderSession, error) {
	pat := qoderPAT(auth)
	if pat == "" {
		return nil, statusErr{code: http.StatusUnauthorized, msg: "missing qoder personal access token"}
	}
	key := qoderSessionKey(auth, pat)
	e.mu.Lock()
	if sess := e.sessions[key]; sess != nil {
		e.mu.Unlock()
		return sess, nil
	}
	e.mu.Unlock()

	machineID, machineToken, machineType := newQoderMachineIdentity()
	identity, err := e.exchangeJobToken(ctx, auth, pat, machineID, machineToken, machineType)
	if err != nil {
		return nil, err
	}
	sess, err := newQoderSession(identity, machineID, machineToken, machineType)
	if err != nil {
		return nil, err
	}
	if err := e.sendHeartbeat(ctx, auth, sess); err != nil {
		log.Debugf("qoder heartbeat failed auth=%s machine=%s error=%v", qoderAuthLabel(auth), machineID, err)
	}

	e.mu.Lock()
	e.sessions[key] = sess
	e.mu.Unlock()
	log.Infof("qoder session ready auth=%s user=%s uid=%s machineType=%s", qoderAuthLabel(auth), identity.Name, identity.UID, machineType)
	return sess, nil
}

// exchangeJobToken follows qoder2api's personal-token-to-job-token flow.
func (e *QoderExecutor) exchangeJobToken(ctx context.Context, auth *cliproxyauth.Auth, pat, machineID, machineToken, machineType string) (qoderAuthIdentity, error) {
	inner := map[string]any{
		"personalToken":      pat,
		"securityOauthToken": "",
		"refreshToken":       "",
		"needRefresh":        false,
		"authInfo":           map[string]any{},
	}
	outer := map[string]any{
		"payload":       mustJSONText(inner),
		"encodeVersion": "1",
	}
	body, err := e.postQoderControl(ctx, auth, qoderControlBaseURL+qoderJobTokenPath, machineID, machineToken, machineType, outer)
	if err != nil {
		return qoderAuthIdentity{}, err
	}
	userID := gjson.GetBytes(body, "id").String()
	return qoderAuthIdentity{
		Name:               gjson.GetBytes(body, "name").String(),
		AID:                userID,
		UID:                userID,
		UserType:           valueOrDefault(gjson.GetBytes(body, "userType").String(), "personal_standard"),
		SecurityOauthToken: gjson.GetBytes(body, "securityOauthToken").String(),
		RefreshToken:       gjson.GetBytes(body, "refreshToken").String(),
	}, nil
}

// sendHeartbeat preserves the control-plane heartbeat behavior from qoder2api.
func (e *QoderExecutor) sendHeartbeat(ctx context.Context, auth *cliproxyauth.Auth, sess *qoderSession) error {
	if sess == nil {
		return nil
	}
	heartbeat := map[string]any{
		"event_time":  time.Now().UnixMilli(),
		"event_type":  "cosy_heartbeat",
		"mid":         sess.machineID,
		"os_arch":     "windows_amd64",
		"os_version":  "Windows",
		"ide_type":    "qodercli",
		"ide_version": qoderVersion,
		"extra_info":  map[string]any{},
	}
	_, err := e.postQoderControl(ctx, auth, qoderControlBaseURL+qoderHeartbeatPath, sess.machineID, sess.machineToken, sess.machineType, heartbeat)
	return err
}

// postQoderControl signs and encodes Qoder control-plane JSON requests.
func (e *QoderExecutor) postQoderControl(ctx context.Context, auth *cliproxyauth.Auth, url, machineID, machineToken, machineType string, payload any) ([]byte, error) {
	plain, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	encoded, err := qoderEncode(plain)
	if err != nil {
		return nil, err
	}
	date := qoderHTTPDate(time.Now())
	reqCtx := ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, strings.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("cosy-machinetoken", machineToken)
	httpReq.Header.Set("cosy-machinetype", machineType)
	httpReq.Header.Set("login-version", "v2")
	httpReq.Header.Set("appcode", qoderAppCode)
	httpReq.Header.Set("accept", "application/json")
	httpReq.Header.Set("accept-encoding", "identity")
	httpReq.Header.Set("cosy-version", qoderVersion)
	httpReq.Header.Set("cosy-clienttype", qoderClientType)
	httpReq.Header.Set("date", date)
	httpReq.Header.Set("signature", qoderControlSignature(date))
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("cosy-machineid", machineID)
	httpReq.Header.Set("user-agent", qoderUserAgent)

	client := helps.NewProxyAwareHTTPClient(reqCtx, e.cfg, auth, qoderControlRequestTimeout)
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, statusErr{code: resp.StatusCode, msg: string(body)}
	}
	return body, nil
}

// buildQoderBody deep-copies the embedded Qoder template and patches request-specific fields.
func buildQoderBody(model, prompt, userType string) ([]byte, string, error) {
	body := qoderBasePromptWithRuntimePlaceholders()
	nowMs := time.Now().UnixMilli()
	requestID := uuid.NewString()
	requestSetID := uuid.NewString()
	sessionID := uuid.NewString()
	businessID := uuid.NewString()
	name := prompt
	if len(name) > 30 {
		name = name[:30]
	}
	if userType == "" {
		userType = "personal_standard"
	}
	var err error
	for _, op := range []struct {
		path  string
		value any
	}{
		{"request_id", requestID},
		{"chat_record_id", requestID},
		{"request_set_id", requestSetID},
		{"session_id", sessionID},
		{"stream", true},
		{"aliyun_user_type", userType},
		{"model_config.key", model},
		{"chat_context.extra.modelConfig.key", model},
		{"business.id", businessID},
		{"business.begin_at", nowMs},
		{"business.name", name},
		{"chat_context.text.text", prompt},
		{"chat_context.extra.originalContent.text", prompt},
	} {
		body, err = sjson.SetBytes(body, op.path, op.value)
		if err != nil {
			return nil, "", err
		}
	}
	body, err = qoderReplaceUserMessages(body, prompt)
	if err != nil {
		return nil, "", err
	}
	modelSource := strings.TrimSpace(gjson.GetBytes(body, "model_config.source").String())
	if modelSource == "" {
		modelSource = qoderDefaultModelSource
	}
	return body, modelSource, nil
}

// qoderReplaceUserMessages keeps template system/tool messages and appends the current user prompt.
func qoderReplaceUserMessages(body []byte, prompt string) ([]byte, error) {
	var messages []json.RawMessage
	gjson.GetBytes(body, "messages").ForEach(func(_, value gjson.Result) bool {
		if !strings.EqualFold(value.Get("role").String(), "user") {
			messages = append(messages, json.RawMessage(value.Raw))
		}
		return true
	})
	userMessage, err := qoderUserMessage(prompt)
	if err != nil {
		return nil, err
	}
	messages = append(messages, userMessage)
	raw, err := json.Marshal(messages)
	if err != nil {
		return nil, err
	}
	return sjson.SetRawBytes(body, "messages", raw)
}

// qoderUserMessage creates the Qoder-side user message shape expected by baseprompt.json.
func qoderUserMessage(prompt string) (json.RawMessage, error) {
	msg := map[string]any{
		"role":    "user",
		"content": "",
		"contents": []map[string]any{{
			"type": "text",
			"text": prompt,
		}},
		"response_meta": map[string]any{
			"id": "",
			"usage": map[string]any{
				"prompt_tokens":     0,
				"completion_tokens": 0,
				"total_tokens":      0,
				"completion_tokens_details": map[string]any{
					"reasoning_tokens": 0,
				},
			},
		},
		"reasoning_content_signature": "",
	}
	raw, err := json.Marshal(msg)
	return json.RawMessage(raw), err
}

// consumeQoderTextStream reads Qoder SSE lines, applies the same idle-stop behavior
// as qoder2api, and passes decoded assistant text deltas to onText.
func consumeQoderTextStream(ctx context.Context, body io.Reader, traceID string, onText func(string)) error {
	lineCh := make(chan string)
	errCh := make(chan error, 1)
	go func() {
		defer close(lineCh)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			line := strings.TrimRight(scanner.Text(), "\r")
			if line != "" {
				lineCh <- line
			}
		}
		if err := scanner.Err(); err != nil {
			errCh <- err
		}
		close(errCh)
	}()
	idle := time.NewTimer(qoderStreamIdleTimeout)
	defer idle.Stop()
	lines := 0
	textChars := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case line, ok := <-lineCh:
			if !ok {
				log.Debugf("qoder stream complete trace=%s reason=eof lines=%d outputChars=%d", traceID, lines, textChars)
				return <-errCh
			}
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(qoderStreamIdleTimeout)
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			lines++
			content := extractQoderContent(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			if content != "" {
				textChars += len(content)
				onText(content)
			}
		case <-idle.C:
			log.Debugf("qoder stream complete trace=%s reason=idle_timeout lines=%d outputChars=%d", traceID, lines, textChars)
			return nil
		}
	}
}

// extractQoderContent parses the native wrapper.body JSON and returns the first delta content.
func extractQoderContent(dataLine string) string {
	inner := gjson.Get(dataLine, "body").String()
	if inner == "" {
		return ""
	}
	var out string
	gjson.Get(inner, "choices").ForEach(func(_, choice gjson.Result) bool {
		out = choice.Get("delta.content").String()
		return out == ""
	})
	return out
}

// qoderOriginalPayload prefers the inbound raw request so prompt extraction sees
// the user's original OpenAI Responses, OpenAI Chat, or Anthropic Messages shape.
func qoderOriginalPayload(req cliproxyexecutor.Request, opts cliproxyexecutor.Options) []byte {
	if len(opts.OriginalRequest) > 0 {
		return opts.OriginalRequest
	}
	return req.Payload
}

// qoderPromptFromPayload converts supported client schemas into Qoder's plain transcript.
func qoderPromptFromPayload(payload []byte, opts cliproxyexecutor.Options) string {
	switch strings.ToLower(strings.TrimSpace(opts.SourceFormat.String())) {
	case "openai-response":
		return promptFromOpenAIResponses(payload)
	case "claude":
		return promptFromAnthropicMessages(payload)
	default:
		return promptFromRoleMessages(gjson.GetBytes(payload, "messages"), "")
	}
}

// promptFromRoleMessages converts role/content arrays into a plain role-labeled transcript.
func promptFromRoleMessages(messages gjson.Result, systemPrefix string) string {
	var sb strings.Builder
	appendPromptBlock(&sb, "system", systemPrefix)
	if messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			role := valueOrDefault(msg.Get("role").String(), "user")
			appendPromptBlock(&sb, role, textFromContent(msg.Get("content")))
			return true
		})
	}
	return strings.TrimSpace(sb.String())
}

// promptFromOpenAIResponses converts Responses API input/instructions into a Qoder transcript.
func promptFromOpenAIResponses(payload []byte) string {
	req := gjson.ParseBytes(payload)
	var sb strings.Builder
	appendPromptBlock(&sb, "system", textFromContent(req.Get("instructions")))
	input := req.Get("input")
	if input.Type == gjson.String {
		appendPromptBlock(&sb, "user", input.String())
	} else if input.IsArray() {
		input.ForEach(func(_, item gjson.Result) bool {
			if item.Get("role").Exists() {
				appendPromptBlock(&sb, valueOrDefault(item.Get("role").String(), "user"), textFromContent(item.Get("content")))
			} else if item.Get("content").Exists() {
				appendPromptBlock(&sb, valueOrDefault(item.Get("type").String(), "input"), textFromContent(item.Get("content")))
			} else {
				appendPromptBlock(&sb, valueOrDefault(item.Get("type").String(), "input"), item.Raw)
			}
			return true
		})
	}
	if strings.TrimSpace(sb.String()) == "" {
		appendPromptBlock(&sb, "user", promptFromRoleMessages(req.Get("messages"), ""))
	}
	return strings.TrimSpace(sb.String())
}

// promptFromAnthropicMessages converts Claude Messages system/messages fields into a transcript.
func promptFromAnthropicMessages(payload []byte) string {
	req := gjson.ParseBytes(payload)
	return promptFromRoleMessages(req.Get("messages"), textFromContent(req.Get("system")))
}

// textFromContent extracts text from strings, OpenAI arrays, and Anthropic content blocks.
func textFromContent(content gjson.Result) string {
	if !content.Exists() || content.Type == gjson.Null {
		return ""
	}
	if content.Type == gjson.String {
		return content.String()
	}
	if content.IsArray() {
		var parts []string
		content.ForEach(func(_, item gjson.Result) bool {
			text := item.Get("text").String()
			if text == "" && item.Get("content").Exists() {
				text = textFromContent(item.Get("content"))
			}
			if text == "" && item.Get("input").Exists() {
				text = textFromContent(item.Get("input"))
			}
			if text == "" && item.Get("type").Exists() && !strings.Contains(item.Get("type").String(), "image") {
				text = item.Raw
			}
			if strings.TrimSpace(text) != "" {
				parts = append(parts, text)
			}
			return true
		})
		return strings.Join(parts, "\n")
	}
	if text := content.Get("text").String(); text != "" {
		return text
	}
	return content.Raw
}

// appendPromptBlock appends a role/content section and skips blank sections.
func appendPromptBlock(sb *strings.Builder, role, content string) {
	content = strings.TrimSpace(content)
	if content == "" {
		return
	}
	if sb.Len() > 0 {
		sb.WriteString("\n\n")
	}
	sb.WriteString(role)
	sb.WriteString(":\n")
	sb.WriteString(content)
}

// qoderBasePromptWithRuntimePlaceholders fills template UUID/time placeholders.
func qoderBasePromptWithRuntimePlaceholders() []byte {
	text := string(qoderBasePromptTemplate)
	for i := 1; i <= 5; i++ {
		text = strings.ReplaceAll(text, fmt.Sprintf("{UUID%d}", i), uuid.NewString())
	}
	text = strings.ReplaceAll(text, "{TIME1}", fmt.Sprint(time.Now().UnixMilli()))
	return []byte(text)
}

// applyQoderStreamHeaders reproduces qoder2api's signed Qoder stream headers.
func applyQoderStreamHeaders(req *http.Request, sess *qoderSession, bearer, cosyDate, model, modelSource string) {
	req.Header.Set("cosy-data-policy", "AGREE")
	req.Header.Set("content-type", "application/json")
	req.Header.Set("cosy-machinetype", sess.machineType)
	req.Header.Set("cosy-clienttype", qoderClientType)
	req.Header.Set("cosy-date", cosyDate)
	req.Header.Set("cosy-user", sess.identity.UID)
	req.Header.Set("cosy-key", sess.cosyKey)
	req.Header.Set("cache-control", "no-cache")
	req.Header.Set("accept", "text/event-stream")
	req.Header.Set("cosy-clientip", qoderClientIP)
	req.Header.Set("authorization", bearer)
	req.Header.Set("accept-encoding", "identity")
	req.Header.Set("cosy-version", qoderVersion)
	req.Header.Set("cosy-machineid", sess.machineID)
	req.Header.Set("cosy-machinetoken", sess.machineToken)
	req.Header.Set("login-version", "v2")
	req.Header.Set("user-agent", qoderUserAgent)
	req.Header.Set("x-model-key", model)
	req.Header.Set("x-model-source", valueOrDefault(modelSource, qoderDefaultModelSource))
}

// newQoderSession encrypts Qoder identity payload and creates bearer signing material.
func newQoderSession(identity qoderAuthIdentity, machineID, machineToken, machineType string) (*qoderSession, error) {
	tempKey := []byte(strings.ReplaceAll(uuid.NewString(), "-", "")[:16])
	cosyKeyBytes, err := qoderRSAEncrypt(tempKey)
	if err != nil {
		return nil, err
	}
	infoPlain, err := json.Marshal(identity)
	if err != nil {
		return nil, err
	}
	infoBytes, err := qoderAESEncrypt(infoPlain, tempKey)
	if err != nil {
		return nil, err
	}
	return &qoderSession{
		identity:     identity,
		machineID:    machineID,
		machineToken: machineToken,
		machineType:  machineType,
		cosyKey:      base64.StdEncoding.EncodeToString(cosyKeyBytes),
		info:         base64.StdEncoding.EncodeToString(infoBytes),
		createdAt:    time.Now(),
	}, nil
}

// buildQoderBearer signs the encoded body and Qoder API path into a COSY bearer token.
func buildQoderBearer(sess *qoderSession, encodedBody, pathWithoutAlgo string) (string, string, error) {
	payload := map[string]string{
		"cosyVersion": qoderVersion,
		"ideVersion":  "",
		"info":        sess.info,
		"requestId":   uuid.NewString(),
		"version":     "v1",
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", "", err
	}
	payloadB64 := base64.StdEncoding.EncodeToString(raw)
	cosyDate := fmt.Sprint(time.Now().Unix())
	sig := qoderMD5Hex(payloadB64 + "\n" + sess.cosyKey + "\n" + cosyDate + "\n" + encodedBody + "\n" + pathWithoutAlgo)
	return "Bearer COSY." + payloadB64 + "." + sig, cosyDate, nil
}

// qoderRSAEncrypt encrypts the temporary AES key with Qoder's server public key.
func qoderRSAEncrypt(tempKey []byte) ([]byte, error) {
	block, _ := pem.Decode([]byte(qoderServerPublicKeyPEM))
	if block == nil {
		return nil, fmt.Errorf("qoder public key decode failed")
	}
	pubAny, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	pub, ok := pubAny.(*rsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("qoder public key is not RSA")
	}
	return rsa.EncryptPKCS1v15(rand.Reader, pub, tempKey)
}

// qoderAESEncrypt encrypts the identity JSON with AES-CBC and PKCS#7 padding.
func qoderAESEncrypt(plain, key []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	padded := pkcs7Pad(plain, aes.BlockSize)
	out := make([]byte, len(padded))
	cipher.NewCBCEncrypter(block, key).CryptBlocks(out, padded)
	return out, nil
}

func pkcs7Pad(data []byte, blockSize int) []byte {
	padding := blockSize - len(data)%blockSize
	out := make([]byte, len(data)+padding)
	copy(out, data)
	for i := len(data); i < len(out); i++ {
		out[i] = byte(padding)
	}
	return out
}

// qoderControlSignature signs Qoder control-plane requests with the COSY app secret.
func qoderControlSignature(date string) string {
	return qoderMD5Hex(qoderAppCode + "&" + qoderSecret + "&" + date)
}

func qoderHTTPDate(t time.Time) string {
	return t.UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT")
}

func qoderMD5Hex(s string) string {
	sum := md5.Sum([]byte(s))
	return hex.EncodeToString(sum[:])
}

const qoderCustomAlphabet = "_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!"
const qoderStdAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

// qoderEncode applies Qoder's base64 rotation and custom alphabet substitution.
func qoderEncode(plaintext []byte) (string, error) {
	std := base64.StdEncoding.EncodeToString(plaintext)
	n := len(std)
	a := n / 3
	rearranged := std[n-a:] + std[a:n-a] + std[:a]
	var sb strings.Builder
	sb.Grow(n)
	for _, r := range rearranged {
		if r == '=' {
			sb.WriteByte('$')
			continue
		}
		idx := strings.IndexRune(qoderStdAlphabet, r)
		if idx < 0 {
			return "", fmt.Errorf("qoder encode: char out of alphabet: %q", r)
		}
		sb.WriteByte(qoderCustomAlphabet[idx])
	}
	return sb.String(), nil
}

// qoderDecode reverses qoderEncode and is used by tests and diagnostics.
func qoderDecode(encoded string) ([]byte, error) {
	n := len(encoded)
	var mapped strings.Builder
	mapped.Grow(n)
	for _, r := range encoded {
		if r == '$' {
			mapped.WriteByte('=')
			continue
		}
		idx := strings.IndexRune(qoderCustomAlphabet, r)
		if idx < 0 {
			return nil, fmt.Errorf("qoder decode: char out of custom alphabet: %q", r)
		}
		mapped.WriteByte(qoderStdAlphabet[idx])
	}
	stdMapped := mapped.String()
	a := n / 3
	std := stdMapped[n-a:] + stdMapped[a:n-a] + stdMapped[:a]
	return base64.StdEncoding.DecodeString(std)
}

func qoderPAT(auth *cliproxyauth.Auth) string {
	if auth == nil || auth.Attributes == nil {
		return ""
	}
	for _, key := range []string{"api_key", "pat", "personal_token", "personalToken"} {
		if val := strings.TrimSpace(auth.Attributes[key]); val != "" {
			return val
		}
	}
	return ""
}

func qoderSessionKey(auth *cliproxyauth.Auth, pat string) string {
	id := "default"
	if auth != nil && strings.TrimSpace(auth.ID) != "" {
		id = strings.TrimSpace(auth.ID)
	}
	sum := md5.Sum([]byte(pat))
	return id + ":" + hex.EncodeToString(sum[:])
}

func qoderAuthLabel(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	if auth.Label != "" {
		return auth.Label
	}
	return auth.ID
}

func newQoderMachineIdentity() (machineID, machineToken, machineType string) {
	machineID = uuid.NewString()
	seed := uuid.NewString() + uuid.NewString()
	if len(seed) > 50 {
		seed = seed[:50]
	}
	machineToken = base64.RawURLEncoding.EncodeToString([]byte(seed))
	machineType = strings.ReplaceAll(uuid.NewString(), "-", "")[:18]
	return machineID, machineToken, machineType
}

func qoderNormalizeModel(model string) string {
	base := strings.TrimSpace(thinking.ParseSuffix(model).ModelName)
	if base == "" {
		return qoderDefaultModel
	}
	return base
}

func qoderShortID() string {
	return strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
}

func valueOrDefault(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func mustJSONText(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func qoderNonStreamResponse(format, model, text string) []byte {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "openai-response":
		return mustJSON(qoderResponseCompleted("resp_"+qoderShortID(), time.Now().Unix(), model, text))
	case "claude":
		return mustJSON(qoderAnthropicMessage("msg_"+qoderShortID(), model, text))
	default:
		return mustJSON(qoderChatCompletion("chatcmpl_"+qoderShortID(), time.Now().Unix(), model, text))
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func qoderZeroOpenAIUsage() map[string]any {
	return map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}

func qoderZeroResponsesUsage() map[string]any {
	return map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}

func qoderZeroAnthropicUsage() map[string]any {
	return map[string]any{"input_tokens": 0, "output_tokens": 0}
}

func qoderChatCompletion(id string, created int64, model, text string) map[string]any {
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": text,
			},
			"finish_reason": "stop",
		}},
		"usage": qoderZeroOpenAIUsage(),
	}
}

func qoderChatChunk(id string, created int64, model, content string, done bool) []byte {
	choice := map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": nil}
	if done {
		choice["finish_reason"] = "stop"
	} else {
		choice["delta"] = map[string]any{"role": "assistant", "content": content}
	}
	return mustJSON(map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{choice},
	})
}

func qoderResponseStart(id string, created int64, model string) map[string]any {
	return map[string]any{"id": id, "object": "response", "created_at": created, "status": "in_progress", "model": model, "output": []any{}, "usage": qoderZeroResponsesUsage()}
}

func qoderResponseCompleted(id string, created int64, model, text string) map[string]any {
	return map[string]any{
		"id":           id,
		"object":       "response",
		"created_at":   created,
		"status":       "completed",
		"completed_at": created,
		"model":        model,
		"output": []map[string]any{{
			"id":     "msg_" + qoderShortID(),
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        text,
				"annotations": []any{},
			}},
		}},
		"usage": qoderZeroResponsesUsage(),
	}
}

func qoderAnthropicMessage(id, model, text string) map[string]any {
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       []map[string]any{{"type": "text", "text": text}},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage":         qoderZeroAnthropicUsage(),
	}
}

type qoderStreamEmitter struct {
	format   string
	model    string
	id       string
	created  int64
	itemID   string
	partID   string
	seq      int
	fullText strings.Builder
}

func newQoderStreamEmitter(format, model string) *qoderStreamEmitter {
	prefix := "chatcmpl_"
	if format == "openai-response" {
		prefix = "resp_"
	} else if format == "claude" {
		prefix = "msg_"
	}
	return &qoderStreamEmitter{
		format:  format,
		model:   model,
		id:      prefix + qoderShortID(),
		created: time.Now().Unix(),
		itemID:  "msg_" + qoderShortID(),
		partID:  "part_" + qoderShortID(),
		seq:     1,
	}
}

func (e *qoderStreamEmitter) start() [][]byte {
	switch e.format {
	case "openai-response":
		return [][]byte{
			qoderResponsesData(e.responseStateEvent("response.created", qoderResponseStart(e.id, e.created, e.model))),
			qoderResponsesData(e.responseEvent("response.output_item.added", map[string]any{"output_index": 0, "item": map[string]any{"id": e.itemID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}})),
			qoderResponsesData(e.responseEvent("response.content_part.added", map[string]any{"item_id": e.itemID, "output_index": 0, "content_index": 0, "part": map[string]any{"id": e.partID, "type": "output_text", "text": "", "annotations": []any{}}})),
		}
	case "claude":
		return [][]byte{
			qoderAnthropicEvent("message_start", map[string]any{"type": "message_start", "message": qoderAnthropicMessage(e.id, e.model, "")}),
			qoderAnthropicEvent("content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""}}),
		}
	default:
		return nil
	}
}

func (e *qoderStreamEmitter) delta(delta string) [][]byte {
	e.fullText.WriteString(delta)
	switch e.format {
	case "openai-response":
		return [][]byte{qoderResponsesData(e.responseEvent("response.output_text.delta", map[string]any{"response_id": e.id, "item_id": e.itemID, "output_index": 0, "content_index": 0, "delta": delta}))}
	case "claude":
		return [][]byte{qoderAnthropicEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": delta}})}
	default:
		return [][]byte{qoderChatChunk(e.id, e.created, e.model, delta, false)}
	}
}

func (e *qoderStreamEmitter) done() [][]byte {
	text := e.fullText.String()
	switch e.format {
	case "openai-response":
		return [][]byte{
			qoderResponsesData(e.responseEvent("response.output_text.done", map[string]any{"response_id": e.id, "item_id": e.itemID, "output_index": 0, "content_index": 0, "delta": "", "text": text})),
			qoderResponsesData(e.responseEvent("response.content_part.done", map[string]any{"response_id": e.id, "item_id": e.itemID, "output_index": 0, "content_index": 0, "part": map[string]any{"id": e.partID, "type": "output_text", "text": text, "annotations": []any{}}})),
			qoderResponsesData(e.responseEvent("response.output_item.done", map[string]any{"output_index": 0, "item": map[string]any{"id": e.itemID, "type": "message", "status": "completed", "role": "assistant", "content": []map[string]any{{"type": "output_text", "text": text, "annotations": []any{}}}}})),
			qoderResponsesData(e.responseStateEvent("response.completed", qoderResponseCompleted(e.id, e.created, e.model, text))),
			[]byte("data: [DONE]\n\n"),
		}
	case "claude":
		return [][]byte{
			qoderAnthropicEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0, "content_block": map[string]any{"type": "text", "text": text, "annotations": []any{}}}),
			qoderAnthropicEvent("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 0}}),
			qoderAnthropicEvent("message_stop", map[string]any{"type": "message_stop"}),
		}
	default:
		return [][]byte{qoderChatChunk(e.id, e.created, e.model, "", true)}
	}
}

func (e *qoderStreamEmitter) responseEvent(eventType string, payload map[string]any) map[string]any {
	payload["type"] = eventType
	payload["sequence_number"] = e.seq
	e.seq++
	return payload
}

func (e *qoderStreamEmitter) responseStateEvent(eventType string, response map[string]any) map[string]any {
	event := map[string]any{"type": eventType, "sequence_number": e.seq, "response": response}
	e.seq++
	return event
}

func qoderResponsesData(payload map[string]any) []byte {
	return []byte("data: " + string(mustJSON(payload)) + "\n\n")
}

func qoderAnthropicEvent(name string, payload map[string]any) []byte {
	return []byte("event: " + name + "\ndata: " + string(mustJSON(payload)) + "\n\n")
}
