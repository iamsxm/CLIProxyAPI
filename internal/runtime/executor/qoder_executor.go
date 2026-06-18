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
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
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
	qoderStreamIdleTimeout     = 120 * time.Second
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

// RequestToFormat tells the shared execution pipeline to translate inbound
// Claude/Gemini/Responses payloads into OpenAI Chat shape before Qoder signing.
func (e *QoderExecutor) RequestToFormat(cliproxyexecutor.Request, cliproxyexecutor.Options) sdktranslator.Format {
	return sdktranslator.FormatOpenAI
}

// Execute converts the caller payload to a Qoder prompt, collects Qoder SSE deltas,
// and returns a non-streaming response matching the caller's original protocol.
func (e *QoderExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	if opts.Alt == "responses/compact" {
		return resp, statusErr{code: http.StatusNotImplemented, msg: "/responses/compact not supported by qoder"}
	}
	baseModel := qoderNormalizeModel(req.Model)
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	payload := qoderRequestPayload(req, opts)
	prompt := qoderPromptForRequest(req, opts)
	traceID := qoderShortID()
	log.Infof("qoder request start trace=%s stream=false model=%s promptChars=%d", traceID, baseModel, len(prompt))

	completion := newQoderCompletion()
	begin := time.Now()
	if err = e.streamQoderDeltas(ctx, auth, baseModel, prompt, payload, traceID, func(delta qoderDelta) {
		completion.Add(delta)
	}); err != nil {
		return resp, err
	}
	log.Infof("qoder request done trace=%s stream=false model=%s outputChars=%d toolCalls=%d costMs=%d", traceID, baseModel, len(completion.Content), len(completion.ToolCalls), time.Since(begin).Milliseconds())
	reporter.EnsurePublished(ctx)
	return cliproxyexecutor.Response{Payload: qoderNonStreamResponse(cliproxyexecutor.ResponseFormatOrSource(opts).String(), baseModel, completion)}, nil
}

// ExecuteStream converts Qoder deltas into the streaming envelope expected by
// OpenAI chat completions, OpenAI Responses, or Anthropic Messages handlers.
func (e *QoderExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := qoderNormalizeModel(req.Model)
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	payload := qoderRequestPayload(req, opts)
	prompt := qoderPromptForRequest(req, opts)
	traceID := qoderShortID()
	log.Infof("qoder request start trace=%s stream=true format=%s model=%s promptChars=%d", traceID, opts.SourceFormat.String(), baseModel, len(prompt))

	httpResp, err := e.openQoderStream(ctx, auth, baseModel, prompt, payload, traceID)
	if err != nil {
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go e.forwardQoderStream(ctx, httpResp, out, cliproxyexecutor.ResponseFormatOrSource(opts).String(), baseModel, traceID)
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

// streamQoderDeltas opens the native Qoder SSE endpoint and emits structured
// assistant deltas so reasoning and tool calls are not lost.
func (e *QoderExecutor) streamQoderDeltas(ctx context.Context, auth *cliproxyauth.Auth, model, prompt string, payload []byte, traceID string, onDelta func(qoderDelta)) error {
	httpResp, err := e.openQoderStream(ctx, auth, model, prompt, payload, traceID)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := httpResp.Body.Close(); closeErr != nil {
			log.Debugf("qoder close stream body trace=%s error=%v", traceID, closeErr)
		}
	}()
	return consumeQoderDeltaStream(ctx, httpResp.Body, traceID, onDelta)
}

// streamQoderText preserves the old text-only helper for focused tests and
// callers that only care about assistant content.
func (e *QoderExecutor) streamQoderText(ctx context.Context, auth *cliproxyauth.Auth, model, prompt string, payload []byte, traceID string, onText func(string)) error {
	return e.streamQoderDeltas(ctx, auth, model, prompt, payload, traceID, func(delta qoderDelta) {
		if delta.Content != "" {
			onText(delta.Content)
		}
	})
}

// openQoderStream builds the signed native Qoder request using the current bearer session.
func (e *QoderExecutor) openQoderStream(ctx context.Context, auth *cliproxyauth.Auth, model, prompt string, payload []byte, traceID string) (*http.Response, error) {
	sess, err := e.getOrCreateSession(ctx, auth)
	if err != nil {
		return nil, err
	}
	body, modelSource, err := buildQoderBodyFromPayload(model, prompt, sess.identity.UserType, payload)
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
	err := consumeQoderDeltaStream(ctx, httpResp.Body, traceID, func(delta qoderDelta) {
		chunks++
		chars += len(delta.Content)
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

// buildQoderBodyFromPayload preserves qoder2api's structured message handling:
// OpenAI-style messages/tools are forwarded to Qoder instead of flattening the
// entire conversation into one user prompt.
func buildQoderBodyFromPayload(model, prompt, userType string, payload []byte) ([]byte, string, error) {
	body, modelSource, err := buildQoderBody(model, prompt, userType)
	if err != nil {
		return nil, "", err
	}
	if !gjson.ValidBytes(payload) || !gjson.GetBytes(payload, "messages").IsArray() {
		return body, modelSource, nil
	}
	messages, err := qoderBuildMessagesFromPayload(body, payload, prompt)
	if err != nil {
		return nil, "", err
	}
	rawMessages, err := json.Marshal(messages)
	if err != nil {
		return nil, "", err
	}
	body, err = sjson.SetRawBytes(body, "messages", rawMessages)
	if err != nil {
		return nil, "", err
	}
	if tools := gjson.GetBytes(payload, "tools"); tools.IsArray() && len(tools.Array()) > 0 {
		body, err = sjson.SetRawBytes(body, "tools", []byte(tools.Raw))
		if err != nil {
			return nil, "", err
		}
	}
	return body, modelSource, nil
}

// qoderBuildMessagesFromPayload converts OpenAI Chat messages to the Qoder
// message schema while keeping the embedded template system prompt when the
// client did not provide its own system message.
func qoderBuildMessagesFromPayload(templateBody, payload []byte, prompt string) ([]json.RawMessage, error) {
	incoming := gjson.GetBytes(payload, "messages")
	toolsEnabled := gjson.GetBytes(payload, "tools").IsArray() && len(gjson.GetBytes(payload, "tools").Array()) > 0
	messages := make([]json.RawMessage, 0, len(incoming.Array())+4)
	if !qoderMessagesHaveRole(incoming, "system") {
		gjson.GetBytes(templateBody, "messages").ForEach(func(_, value gjson.Result) bool {
			if strings.EqualFold(value.Get("role").String(), "system") {
				messages = append(messages, json.RawMessage(value.Raw))
			}
			return true
		})
	}
	incoming.ForEach(func(_, value gjson.Result) bool {
		if converted := qoderConvertIncomingMessage(value, toolsEnabled); len(converted) > 0 {
			messages = append(messages, converted)
		}
		return true
	})
	if len(messages) == 0 && strings.TrimSpace(prompt) != "" {
		if msg, err := qoderUserMessageWithImages(prompt, nil); err == nil {
			messages = append(messages, msg)
		}
	}
	return messages, nil
}

func qoderMessagesHaveRole(messages gjson.Result, role string) bool {
	found := false
	if !messages.IsArray() {
		return false
	}
	messages.ForEach(func(_, msg gjson.Result) bool {
		if strings.EqualFold(msg.Get("role").String(), role) {
			found = true
			return false
		}
		return true
	})
	return found
}

func qoderConvertIncomingMessage(msg gjson.Result, toolsEnabled bool) json.RawMessage {
	role := valueOrDefault(msg.Get("role").String(), "user")
	text := qoderNormalizeMessageText(msg)
	switch strings.ToLower(role) {
	case "user":
		if strings.TrimSpace(text) == "" && len(qoderExtractImageParts(msg)) == 0 {
			return nil
		}
		raw, _ := qoderUserMessageWithImages(text, qoderExtractImageParts(msg))
		return raw
	case "assistant":
		if toolCalls := msg.Get("tool_calls"); toolCalls.IsArray() && len(toolCalls.Array()) > 0 {
			if toolsEnabled {
				raw, _ := qoderStructuredMessage(role, text, map[string]any{"tool_calls": json.RawMessage(toolCalls.Raw)})
				return raw
			}
			text = joinNonEmptySections(text, "Tool calls:\n"+toolCalls.Raw)
		}
		if toolsEnabled {
			if parsed := qoderParseToolCallsText(text); len(parsed) > 0 {
				raw, _ := qoderStructuredMessage(role, "", map[string]any{"tool_calls": qoderToolCallsForMessage(parsed)})
				return raw
			}
		}
		if strings.TrimSpace(text) == "" {
			return nil
		}
		raw, _ := qoderStructuredMessage(role, text, nil)
		return raw
	case "tool":
		if toolsEnabled {
			extras := map[string]any{}
			if name := msg.Get("name").String(); name != "" {
				extras["name"] = name
			}
			if id := msg.Get("tool_call_id").String(); id != "" {
				extras["tool_call_id"] = id
			}
			raw, _ := qoderStructuredMessage(role, text, extras)
			return raw
		}
		raw, _ := qoderUserMessageWithImages(qoderRenderToolResult(msg, text), nil)
		return raw
	default:
		if strings.TrimSpace(text) == "" {
			return nil
		}
		raw, _ := qoderStructuredMessage(role, text, nil)
		return raw
	}
}

func qoderNormalizeMessageText(msg gjson.Result) string {
	text := textFromContent(msg.Get("content"))
	if strings.TrimSpace(text) == "" {
		text = textFromContent(msg.Get("contents"))
	}
	return text
}

func qoderExtractImageParts(msg gjson.Result) []map[string]any {
	content := msg.Get("content")
	if !content.IsArray() {
		return nil
	}
	parts := make([]map[string]any, 0)
	content.ForEach(func(_, item gjson.Result) bool {
		itemType := item.Get("type").String()
		if itemType != "image_url" && itemType != "input_image" {
			return true
		}
		imageURL := item.Get("image_url")
		if imageURL.Type == gjson.String && imageURL.String() != "" {
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL.String()}})
			return true
		}
		if imageURL.IsObject() && imageURL.Get("url").String() != "" {
			var obj map[string]any
			if err := json.Unmarshal([]byte(imageURL.Raw), &obj); err == nil {
				parts = append(parts, map[string]any{"type": "image_url", "image_url": obj})
			}
			return true
		}
		if url := item.Get("url").String(); url != "" {
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
		}
		return true
	})
	return parts
}

func qoderUserMessageWithImages(text string, images []map[string]any) (json.RawMessage, error) {
	contents := make([]map[string]any, 0, len(images)+1)
	if text != "" {
		contents = append(contents, map[string]any{"type": "text", "text": text})
	}
	contents = append(contents, images...)
	if len(contents) == 0 {
		contents = append(contents, map[string]any{"type": "text", "text": ""})
	}
	msg := map[string]any{
		"role":                        "user",
		"content":                     "",
		"contents":                    contents,
		"response_meta":               qoderBlankResponseMeta(),
		"reasoning_content_signature": "",
	}
	raw, err := json.Marshal(msg)
	return json.RawMessage(raw), err
}

func qoderStructuredMessage(role, text string, extras map[string]any) (json.RawMessage, error) {
	msg := map[string]any{
		"role":                        role,
		"content":                     text,
		"response_meta":               qoderBlankResponseMeta(),
		"reasoning_content_signature": "",
	}
	for key, value := range extras {
		msg[key] = value
	}
	raw, err := json.Marshal(msg)
	return json.RawMessage(raw), err
}

func qoderBlankResponseMeta() map[string]any {
	return map[string]any{
		"id": "",
		"usage": map[string]any{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
			"completion_tokens_details": map[string]any{
				"reasoning_tokens": 0,
			},
			"prompt_tokens_details": map[string]any{
				"cached_tokens": 0,
			},
		},
	}
}

func qoderRenderToolResult(msg gjson.Result, text string) string {
	label := "Tool result"
	if name := msg.Get("name").String(); name != "" {
		label += " (" + name + ")"
	}
	if id := msg.Get("tool_call_id").String(); id != "" {
		label += " [" + id + "]"
	}
	if strings.TrimSpace(text) == "" {
		return label
	}
	return label + ":\n" + text
}

func joinNonEmptySections(first, second string) string {
	first = strings.TrimSpace(first)
	second = strings.TrimSpace(second)
	if first == "" {
		return second
	}
	if second == "" {
		return first
	}
	return first + "\n\n" + second
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

type qoderToolCallDelta struct {
	Index     int
	ID        string
	Type      string
	Name      string
	Arguments string
}

type qoderDelta struct {
	Role      string
	Content   string
	Reasoning string
	ToolCalls []qoderToolCallDelta
}

// consumeQoderTextStream reads Qoder SSE lines, applies the same idle-stop
// behavior as qoder2api, and passes decoded assistant text deltas to onText.
func consumeQoderTextStream(ctx context.Context, body io.Reader, traceID string, onText func(string)) error {
	return consumeQoderDeltaStream(ctx, body, traceID, func(delta qoderDelta) {
		if delta.Content != "" {
			onText(delta.Content)
		}
	})
}

// consumeQoderDeltaStream reads Qoder SSE lines and passes structured deltas to
// the caller. Qoder wraps OpenAI-compatible chunks inside body JSON strings.
func consumeQoderDeltaStream(ctx context.Context, body io.Reader, traceID string, onDelta func(qoderDelta)) error {
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
			delta, errDelta := extractQoderDelta(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			if errDelta != nil {
				return errDelta
			}
			if !delta.Empty() {
				textChars += len(delta.Content)
				onDelta(delta)
			}
		case <-idle.C:
			log.Debugf("qoder stream complete trace=%s reason=idle_timeout lines=%d outputChars=%d", traceID, lines, textChars)
			return nil
		}
	}
}

// extractQoderContent parses the native wrapper.body JSON and returns the first delta content.
func extractQoderContent(dataLine string) string {
	delta, _ := extractQoderDelta(dataLine)
	return delta.Content
}

// extractQoderDelta parses a Qoder SSE data payload and surfaces quota-like
// upstream errors so the auth scheduler can fail over to another PAT.
func extractQoderDelta(dataLine string) (qoderDelta, error) {
	inner := gjson.Get(dataLine, "body").String()
	if inner == "" {
		return qoderDelta{}, nil
	}
	if code, msg := gjson.Get(inner, "code").String(), gjson.Get(inner, "message").String(); code != "" && msg != "" && !gjson.Get(inner, "choices").Exists() {
		if qoderIsQuotaExhaustedError(code) || qoderIsQuotaExhaustedError(msg) {
			return qoderDelta{}, statusErr{code: http.StatusTooManyRequests, msg: "qoder quota exhausted: " + msg}
		}
		log.Warnf("qoder upstream error code=%s message=%s", code, msg)
		return qoderDelta{}, nil
	}
	var out qoderDelta
	gjson.Get(inner, "choices").ForEach(func(_, choice gjson.Result) bool {
		choiceDelta := choice.Get("delta")
		if role := choiceDelta.Get("role").String(); role != "" {
			out.Role = role
		}
		if content := choiceDelta.Get("content"); content.Exists() && content.Type != gjson.Null {
			out.Content += content.String()
		}
		if reasoning := qoderReasoningFromDelta(choiceDelta); reasoning != "" {
			out.Reasoning += reasoning
		}
		if toolCalls := choiceDelta.Get("tool_calls"); toolCalls.IsArray() {
			out.ToolCalls = append(out.ToolCalls, qoderParseToolCallDeltas(toolCalls)...)
		}
		return out.Empty()
	})
	return out, nil
}

func (d qoderDelta) Empty() bool {
	return d.Role == "" && d.Content == "" && d.Reasoning == "" && len(d.ToolCalls) == 0
}

func qoderReasoningFromDelta(delta gjson.Result) string {
	if reasoning := delta.Get("reasoning_content"); reasoning.Exists() && reasoning.Type != gjson.Null {
		return reasoning.String()
	}
	if reasoning := delta.Get("reasoning"); reasoning.Exists() && reasoning.Type != gjson.Null {
		return reasoning.String()
	}
	return ""
}

// qoderParseToolCallDeltas keeps the OpenAI stream shape that Qoder returns so
// downstream clients receive function-call fragments in the same order.
func qoderParseToolCallDeltas(toolCalls gjson.Result) []qoderToolCallDelta {
	out := make([]qoderToolCallDelta, 0, len(toolCalls.Array()))
	toolCalls.ForEach(func(_, tc gjson.Result) bool {
		index := int(tc.Get("index").Int())
		if !tc.Get("index").Exists() {
			index = len(out)
		}
		callType := valueOrDefault(tc.Get("type").String(), "function")
		out = append(out, qoderToolCallDelta{
			Index:     index,
			ID:        tc.Get("id").String(),
			Type:      callType,
			Name:      tc.Get("function.name").String(),
			Arguments: tc.Get("function.arguments").String(),
		})
		return true
	})
	return out
}

type qoderToolCall struct {
	Index     int
	ID        string
	Type      string
	Name      string
	Arguments string
}

type qoderToolCallAccumulator struct {
	order []int
	calls map[int]*qoderToolCall
}

func newQoderToolCallAccumulator() *qoderToolCallAccumulator {
	return &qoderToolCallAccumulator{calls: map[int]*qoderToolCall{}}
}

// Add merges OpenAI-compatible streaming tool call fragments by index. Qoder
// can split id/name and arguments across multiple SSE chunks.
func (a *qoderToolCallAccumulator) Add(delta qoderToolCallDelta) {
	if a == nil {
		return
	}
	call := a.calls[delta.Index]
	if call == nil {
		call = &qoderToolCall{Index: delta.Index, Type: "function"}
		a.calls[delta.Index] = call
		a.order = append(a.order, delta.Index)
	}
	if delta.ID != "" {
		call.ID = delta.ID
	}
	if delta.Type != "" {
		call.Type = delta.Type
	}
	if delta.Name != "" {
		call.Name = delta.Name
	}
	if delta.Arguments != "" {
		call.Arguments += delta.Arguments
	}
}

func (a *qoderToolCallAccumulator) Final() []qoderToolCall {
	if a == nil || len(a.order) == 0 {
		return nil
	}
	out := make([]qoderToolCall, 0, len(a.order))
	for _, index := range a.order {
		call := a.calls[index]
		if call == nil {
			continue
		}
		if call.ID == "" && call.Name == "" && call.Arguments == "" {
			continue
		}
		if call.ID == "" {
			call.ID = fmt.Sprintf("call_%d", index)
		}
		if call.Type == "" {
			call.Type = "function"
		}
		out = append(out, *call)
	}
	return out
}

type qoderCompletion struct {
	Content   string
	Reasoning string
	ToolCalls []qoderToolCall
	tools     *qoderToolCallAccumulator
}

func newQoderCompletion() *qoderCompletion {
	return &qoderCompletion{tools: newQoderToolCallAccumulator()}
}

// Add accumulates a Qoder delta into a complete assistant message, preserving
// text, reasoning text, and function-call fragments.
func (c *qoderCompletion) Add(delta qoderDelta) {
	if c == nil {
		return
	}
	c.Content += delta.Content
	c.Reasoning += delta.Reasoning
	for _, toolCall := range delta.ToolCalls {
		c.tools.Add(toolCall)
	}
	c.ToolCalls = c.tools.Final()
}

func qoderIsQuotaExhaustedError(message string) bool {
	msg := strings.ToLower(strings.TrimSpace(message))
	if msg == "" {
		return false
	}
	return strings.Contains(msg, "quota") ||
		strings.Contains(msg, "exhausted") ||
		strings.Contains(msg, "limit exceeded") ||
		strings.Contains(msg, "daily limit") ||
		strings.Contains(msg, "daily_limit") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "rate_limit") ||
		strings.Contains(msg, "too many") ||
		strings.Contains(msg, "insufficient") ||
		strings.Contains(msg, "429") ||
		(strings.Contains(msg, "403") && (strings.Contains(msg, "balance") || strings.Contains(msg, "credit")))
}

// qoderOriginalPayload prefers the inbound raw request so prompt extraction sees
// the user's original OpenAI Responses, OpenAI Chat, or Anthropic Messages shape.
func qoderOriginalPayload(req cliproxyexecutor.Request, opts cliproxyexecutor.Options) []byte {
	if len(opts.OriginalRequest) > 0 {
		return opts.OriginalRequest
	}
	return req.Payload
}

// qoderRequestPayload prefers the framework-translated OpenAI Chat payload for
// Qoder body construction, with the original request as a compatibility fallback.
func qoderRequestPayload(req cliproxyexecutor.Request, opts cliproxyexecutor.Options) []byte {
	if gjson.ValidBytes(req.Payload) && gjson.GetBytes(req.Payload, "messages").IsArray() {
		return req.Payload
	}
	return qoderOriginalPayload(req, opts)
}

// qoderPromptForRequest mirrors qoder2api: chat_context text is the latest user
// prompt, while the full multi-turn conversation is carried in messages[].
func qoderPromptForRequest(req cliproxyexecutor.Request, opts cliproxyexecutor.Options) string {
	if prompt := qoderLatestUserPrompt(qoderRequestPayload(req, opts)); strings.TrimSpace(prompt) != "" {
		return prompt
	}
	return qoderPromptFromPayload(qoderOriginalPayload(req, opts), opts)
}

func qoderLatestUserPrompt(payload []byte) string {
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() {
		return ""
	}
	arr := messages.Array()
	for i := len(arr) - 1; i >= 0; i-- {
		if strings.EqualFold(arr[i].Get("role").String(), "user") {
			if text := qoderNormalizeMessageText(arr[i]); strings.TrimSpace(text) != "" {
				return text
			}
		}
	}
	return ""
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
	switch base {
	case "Qwen3.7-Max":
		return "qmodel_latest"
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

func qoderNonStreamResponse(format, model string, completion *qoderCompletion) []byte {
	completion = qoderPrepareCompletionForResponse(completion)
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "openai-response":
		return mustJSON(qoderResponseCompleted("resp_"+qoderShortID(), time.Now().Unix(), model, completion))
	case "claude":
		return mustJSON(qoderAnthropicMessage("msg_"+qoderShortID(), model, completion))
	default:
		return mustJSON(qoderChatCompletion("chatcmpl_"+qoderShortID(), time.Now().Unix(), model, completion))
	}
}

func qoderTextCompletion(text string) *qoderCompletion {
	completion := newQoderCompletion()
	completion.Content = text
	return completion
}

func qoderEnsureCompletion(completion *qoderCompletion) *qoderCompletion {
	if completion == nil {
		return newQoderCompletion()
	}
	return completion
}

// qoderPrepareCompletionForResponse mirrors qoder2api's textual tool-call
// fallback: "Tool calls: [...]" is converted back to structured function calls.
func qoderPrepareCompletionForResponse(completion *qoderCompletion) *qoderCompletion {
	completion = qoderEnsureCompletion(completion)
	if len(completion.ToolCalls) > 0 {
		return completion
	}
	parsed := qoderParseToolCallsText(completion.Content)
	if len(parsed) == 0 {
		return completion
	}
	return &qoderCompletion{
		Content:   "",
		Reasoning: completion.Reasoning,
		ToolCalls: parsed,
		tools:     completion.tools,
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

func qoderChatCompletion(id string, created int64, model string, completion *qoderCompletion) map[string]any {
	completion = qoderEnsureCompletion(completion)
	message := map[string]any{
		"role":    "assistant",
		"content": completion.Content,
	}
	if completion.Reasoning != "" {
		message["reasoning_content"] = completion.Reasoning
	}
	if len(completion.ToolCalls) > 0 {
		message["tool_calls"] = qoderOpenAIToolCalls(completion.ToolCalls)
	}
	finishReason := "stop"
	if len(completion.ToolCalls) > 0 {
		finishReason = "tool_calls"
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": qoderZeroOpenAIUsage(),
	}
}

func qoderOpenAIToolCalls(toolCalls []qoderToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(toolCalls))
	for i, call := range toolCalls {
		index := call.Index
		if index < 0 {
			index = i
		}
		out = append(out, map[string]any{
			"index": index,
			"id":    valueOrDefault(call.ID, fmt.Sprintf("call_%d", index)),
			"type":  valueOrDefault(call.Type, "function"),
			"function": map[string]any{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}
	return out
}

func qoderToolCallsForMessage(toolCalls []qoderToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(toolCalls))
	for _, call := range toolCalls {
		out = append(out, map[string]any{
			"id":   call.ID,
			"type": valueOrDefault(call.Type, "function"),
			"function": map[string]any{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}
	return out
}

// qoderParseToolCallsText parses qoder2api-compatible assistant text fallback:
// Tool calls: followed by a JSON array, optionally wrapped in a fenced block.
func qoderParseToolCallsText(text string) []qoderToolCall {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "Tool calls:") {
		return nil
	}
	payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "Tool calls:"))
	if strings.HasPrefix(payload, "```") && strings.HasSuffix(payload, "```") {
		if newline := strings.Index(payload, "\n"); newline >= 0 {
			payload = strings.TrimSpace(payload[newline : len(payload)-3])
		}
	}
	if !strings.HasPrefix(payload, "[") || !gjson.Valid(payload) {
		return nil
	}
	raw := gjson.Parse(payload)
	if !raw.IsArray() {
		return nil
	}
	out := make([]qoderToolCall, 0, len(raw.Array()))
	raw.ForEach(func(_, item gjson.Result) bool {
		name := item.Get("function.name").String()
		args := qoderNormalizeToolArguments(item.Get("function.arguments"))
		if name == "" && args == "" {
			return true
		}
		out = append(out, qoderToolCall{
			Index:     len(out),
			ID:        item.Get("id").String(),
			Type:      valueOrDefault(item.Get("type").String(), "function"),
			Name:      name,
			Arguments: args,
		})
		return true
	})
	return out
}

func qoderNormalizeToolArguments(arguments gjson.Result) string {
	if !arguments.Exists() || arguments.Type == gjson.Null {
		return ""
	}
	if arguments.Type == gjson.String {
		return arguments.String()
	}
	return arguments.Raw
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

func qoderChatDeltaChunk(id string, created int64, model string, delta qoderDelta, done bool) []byte {
	choice := map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": nil}
	if done {
		choice["finish_reason"] = "stop"
	} else {
		payload := map[string]any{"role": "assistant"}
		if delta.Content != "" {
			payload["content"] = delta.Content
		}
		if delta.Reasoning != "" {
			payload["reasoning_content"] = delta.Reasoning
		}
		if len(delta.ToolCalls) > 0 {
			payload["tool_calls"] = qoderOpenAIToolCallDeltas(delta.ToolCalls)
		}
		choice["delta"] = payload
	}
	return mustJSON(map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{choice},
	})
}

func qoderChatDoneChunk(id string, created int64, model string, toolCalls bool) []byte {
	finishReason := "stop"
	if toolCalls {
		finishReason = "tool_calls"
	}
	return mustJSON(map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": finishReason,
		}},
	})
}

func qoderOpenAIToolCallDeltas(toolCalls []qoderToolCallDelta) []map[string]any {
	out := make([]map[string]any, 0, len(toolCalls))
	for _, call := range toolCalls {
		item := map[string]any{
			"index": call.Index,
			"type":  valueOrDefault(call.Type, "function"),
			"function": map[string]any{
				"arguments": call.Arguments,
			},
		}
		if call.ID != "" {
			item["id"] = call.ID
		}
		if call.Name != "" {
			item["function"].(map[string]any)["name"] = call.Name
		}
		out = append(out, item)
	}
	return out
}

func qoderResponseStart(id string, created int64, model string) map[string]any {
	return map[string]any{"id": id, "object": "response", "created_at": created, "status": "in_progress", "model": model, "output": []any{}, "usage": qoderZeroResponsesUsage()}
}

func qoderResponseCompleted(id string, created int64, model string, completion *qoderCompletion) map[string]any {
	completion = qoderEnsureCompletion(completion)
	output := make([]map[string]any, 0, 1+len(completion.ToolCalls))
	if completion.Reasoning != "" {
		output = append(output, map[string]any{
			"id":     "rs_" + qoderShortID(),
			"type":   "reasoning",
			"status": "completed",
			"summary": []map[string]any{{
				"type": "summary_text",
				"text": completion.Reasoning,
			}},
		})
	}
	if completion.Content != "" || len(output) == 0 && len(completion.ToolCalls) == 0 {
		output = append(output, map[string]any{
			"id":     "msg_" + qoderShortID(),
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        completion.Content,
				"annotations": []any{},
			}},
		})
	}
	for _, call := range completion.ToolCalls {
		output = append(output, map[string]any{
			"id":        "fc_" + valueOrDefault(call.ID, qoderShortID()),
			"type":      "function_call",
			"status":    "completed",
			"arguments": call.Arguments,
			"call_id":   valueOrDefault(call.ID, fmt.Sprintf("call_%d", call.Index)),
			"name":      call.Name,
		})
	}
	return map[string]any{
		"id":           id,
		"object":       "response",
		"created_at":   created,
		"status":       "completed",
		"completed_at": created,
		"model":        model,
		"output":       output,
		"usage":        qoderZeroResponsesUsage(),
	}
}

func qoderAnthropicMessage(id, model string, completion *qoderCompletion) map[string]any {
	completion = qoderEnsureCompletion(completion)
	content := make([]map[string]any, 0, 1+len(completion.ToolCalls))
	if completion.Reasoning != "" {
		content = append(content, map[string]any{"type": "thinking", "thinking": completion.Reasoning})
	}
	if completion.Content != "" || len(content) == 0 && len(completion.ToolCalls) == 0 {
		content = append(content, map[string]any{"type": "text", "text": completion.Content})
	}
	for _, call := range completion.ToolCalls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    valueOrDefault(call.ID, fmt.Sprintf("call_%d", call.Index)),
			"name":  call.Name,
			"input": qoderToolInput(call.Arguments),
		})
	}
	stopReason := "end_turn"
	if len(completion.ToolCalls) > 0 {
		stopReason = "tool_use"
	}
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage":         qoderZeroAnthropicUsage(),
	}
}

func qoderToolInput(arguments string) any {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return map[string]any{}
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(trimmed), &obj); err == nil {
		return obj
	}
	var value any
	if err := json.Unmarshal([]byte(trimmed), &value); err == nil {
		return map[string]any{"arguments": value}
	}
	return map[string]any{"arguments": arguments}
}

type qoderStreamEmitter struct {
	format         string
	model          string
	id             string
	created        int64
	itemID         string
	partID         string
	reasoningID    string
	seq            int
	fullText       strings.Builder
	fullReasoning  strings.Builder
	tools          *qoderToolCallAccumulator
	toolItemAdded  map[int]bool
	toolItemDone   map[int]bool
	toolOutputIdx  map[int]int
	toolBlockIndex map[int]int
	claudeIndex    int
	claudeOpen     string
	seenToolCalls  bool
}

func newQoderStreamEmitter(format, model string) *qoderStreamEmitter {
	prefix := "chatcmpl_"
	if format == "openai-response" {
		prefix = "resp_"
	} else if format == "claude" {
		prefix = "msg_"
	}
	return &qoderStreamEmitter{
		format:         format,
		model:          model,
		id:             prefix + qoderShortID(),
		created:        time.Now().Unix(),
		itemID:         "msg_" + qoderShortID(),
		partID:         "part_" + qoderShortID(),
		tools:          newQoderToolCallAccumulator(),
		toolItemAdded:  map[int]bool{},
		toolItemDone:   map[int]bool{},
		toolOutputIdx:  map[int]int{},
		toolBlockIndex: map[int]int{},
		seq:            1,
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
			qoderAnthropicEvent("message_start", map[string]any{"type": "message_start", "message": qoderAnthropicMessage(e.id, e.model, qoderTextCompletion(""))}),
		}
	default:
		return nil
	}
}

func (e *qoderStreamEmitter) delta(delta qoderDelta) [][]byte {
	e.fullText.WriteString(delta.Content)
	e.fullReasoning.WriteString(delta.Reasoning)
	for _, toolCall := range delta.ToolCalls {
		e.seenToolCalls = true
		e.tools.Add(toolCall)
	}
	switch e.format {
	case "openai-response":
		return e.responsesDelta(delta)
	case "claude":
		return e.claudeDelta(delta)
	default:
		return [][]byte{qoderChatDeltaChunk(e.id, e.created, e.model, delta, false)}
	}
}

func (e *qoderStreamEmitter) done() [][]byte {
	text := e.fullText.String()
	completion := newQoderCompletion()
	completion.Content = text
	completion.Reasoning = e.fullReasoning.String()
	completion.ToolCalls = e.tools.Final()
	switch e.format {
	case "openai-response":
		out := [][]byte{
			qoderResponsesData(e.responseEvent("response.output_text.done", map[string]any{"response_id": e.id, "item_id": e.itemID, "output_index": 0, "content_index": 0, "delta": "", "text": text})),
			qoderResponsesData(e.responseEvent("response.content_part.done", map[string]any{"response_id": e.id, "item_id": e.itemID, "output_index": 0, "content_index": 0, "part": map[string]any{"id": e.partID, "type": "output_text", "text": text, "annotations": []any{}}})),
			qoderResponsesData(e.responseEvent("response.output_item.done", map[string]any{"output_index": 0, "item": map[string]any{"id": e.itemID, "type": "message", "status": "completed", "role": "assistant", "content": []map[string]any{{"type": "output_text", "text": text, "annotations": []any{}}}}})),
		}
		out = append(out, e.responsesDone(completion)...)
		out = append(out,
			qoderResponsesData(e.responseStateEvent("response.completed", qoderResponseCompleted(e.id, e.created, e.model, completion))),
			[]byte("data: [DONE]\n\n"),
		)
		return out
	case "claude":
		out := e.claudeCloseOpenBlock()
		out = append(out, e.claudeToolDone(completion.ToolCalls)...)
		stopReason := "end_turn"
		if len(completion.ToolCalls) > 0 {
			stopReason = "tool_use"
		}
		out = append(out,
			qoderAnthropicEvent("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 0}}),
			qoderAnthropicEvent("message_stop", map[string]any{"type": "message_stop"}),
		)
		return out
	default:
		return [][]byte{qoderChatDoneChunk(e.id, e.created, e.model, e.seenToolCalls)}
	}
}

func (e *qoderStreamEmitter) responsesDelta(delta qoderDelta) [][]byte {
	out := make([][]byte, 0, 2+len(delta.ToolCalls)*2)
	if delta.Content != "" {
		out = append(out, qoderResponsesData(e.responseEvent("response.output_text.delta", map[string]any{
			"response_id":   e.id,
			"item_id":       e.itemID,
			"output_index":  0,
			"content_index": 0,
			"delta":         delta.Content,
		})))
	}
	if delta.Reasoning != "" {
		out = append(out, e.responsesReasoningDelta(delta.Reasoning)...)
	}
	for _, toolCall := range delta.ToolCalls {
		out = append(out, e.responsesToolDelta(toolCall)...)
	}
	return out
}

func (e *qoderStreamEmitter) responsesReasoningDelta(text string) [][]byte {
	out := make([][]byte, 0, 3)
	if e.reasoningID == "" {
		e.reasoningID = "rs_" + qoderShortID()
		out = append(out,
			qoderResponsesData(e.responseEvent("response.output_item.added", map[string]any{"output_index": 1, "item": map[string]any{"id": e.reasoningID, "type": "reasoning", "status": "in_progress", "summary": []any{}}})),
			qoderResponsesData(e.responseEvent("response.reasoning_summary_part.added", map[string]any{"item_id": e.reasoningID, "output_index": 1, "summary_index": 0, "part": map[string]any{"type": "summary_text", "text": ""}})),
		)
	}
	out = append(out, qoderResponsesData(e.responseEvent("response.reasoning_summary_text.delta", map[string]any{
		"item_id":       e.reasoningID,
		"output_index":  1,
		"summary_index": 0,
		"delta":         text,
	})))
	return out
}

func (e *qoderStreamEmitter) responsesToolDelta(delta qoderToolCallDelta) [][]byte {
	out := make([][]byte, 0, 2)
	call := e.tools.calls[delta.Index]
	if call == nil {
		return out
	}
	if !e.toolItemAdded[delta.Index] {
		outputIndex := 2 + len(e.toolItemAdded)
		e.toolOutputIdx[delta.Index] = outputIndex
		e.toolItemAdded[delta.Index] = true
		out = append(out, qoderResponsesData(e.responseEvent("response.output_item.added", map[string]any{
			"output_index": outputIndex,
			"item": map[string]any{
				"id":        "fc_" + valueOrDefault(call.ID, fmt.Sprintf("call_%d", delta.Index)),
				"type":      "function_call",
				"status":    "in_progress",
				"arguments": "",
				"call_id":   valueOrDefault(call.ID, fmt.Sprintf("call_%d", delta.Index)),
				"name":      call.Name,
			},
		})))
	}
	if delta.Arguments != "" {
		callID := valueOrDefault(call.ID, fmt.Sprintf("call_%d", delta.Index))
		out = append(out, qoderResponsesData(e.responseEvent("response.function_call_arguments.delta", map[string]any{
			"item_id":      "fc_" + callID,
			"output_index": e.toolOutputIdx[delta.Index],
			"delta":        delta.Arguments,
		})))
	}
	return out
}

func (e *qoderStreamEmitter) responsesDone(completion *qoderCompletion) [][]byte {
	out := make([][]byte, 0, 2+len(completion.ToolCalls)*2)
	if e.reasoningID != "" {
		reasoning := completion.Reasoning
		out = append(out,
			qoderResponsesData(e.responseEvent("response.reasoning_summary_text.done", map[string]any{"item_id": e.reasoningID, "output_index": 1, "summary_index": 0, "text": reasoning})),
			qoderResponsesData(e.responseEvent("response.reasoning_summary_part.done", map[string]any{"item_id": e.reasoningID, "output_index": 1, "summary_index": 0, "part": map[string]any{"type": "summary_text", "text": reasoning}})),
			qoderResponsesData(e.responseEvent("response.output_item.done", map[string]any{"output_index": 1, "item": map[string]any{"id": e.reasoningID, "type": "reasoning", "status": "completed", "summary": []map[string]any{{"type": "summary_text", "text": reasoning}}}})),
		)
	}
	for _, call := range completion.ToolCalls {
		out = append(out, e.responsesToolDone(call)...)
	}
	return out
}

func (e *qoderStreamEmitter) responsesToolDone(call qoderToolCall) [][]byte {
	if e.toolItemDone[call.Index] {
		return nil
	}
	out := make([][]byte, 0, 3)
	if !e.toolItemAdded[call.Index] {
		out = append(out, e.responsesToolDelta(qoderToolCallDelta{Index: call.Index, ID: call.ID, Type: call.Type, Name: call.Name})...)
	}
	callID := valueOrDefault(call.ID, fmt.Sprintf("call_%d", call.Index))
	outputIndex := e.toolOutputIdx[call.Index]
	e.toolItemDone[call.Index] = true
	out = append(out,
		qoderResponsesData(e.responseEvent("response.function_call_arguments.done", map[string]any{"item_id": "fc_" + callID, "output_index": outputIndex, "arguments": call.Arguments})),
		qoderResponsesData(e.responseEvent("response.output_item.done", map[string]any{"output_index": outputIndex, "item": map[string]any{
			"id":        "fc_" + callID,
			"type":      "function_call",
			"status":    "completed",
			"arguments": call.Arguments,
			"call_id":   callID,
			"name":      call.Name,
		}})),
	)
	return out
}

func (e *qoderStreamEmitter) claudeDelta(delta qoderDelta) [][]byte {
	out := make([][]byte, 0, 2)
	if delta.Reasoning != "" {
		out = append(out, e.claudeEnsureBlock("thinking")...)
		out = append(out, qoderAnthropicEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": e.claudeIndex, "delta": map[string]any{"type": "thinking_delta", "thinking": delta.Reasoning}}))
	}
	if delta.Content != "" {
		out = append(out, e.claudeEnsureBlock("text")...)
		out = append(out, qoderAnthropicEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": e.claudeIndex, "delta": map[string]any{"type": "text_delta", "text": delta.Content}}))
	}
	return out
}

func (e *qoderStreamEmitter) claudeEnsureBlock(kind string) [][]byte {
	if e.claudeOpen == kind {
		return nil
	}
	out := e.claudeCloseOpenBlock()
	e.claudeOpen = kind
	contentBlock := map[string]any{"type": kind}
	if kind == "thinking" {
		contentBlock["thinking"] = ""
	} else {
		contentBlock["text"] = ""
	}
	out = append(out, qoderAnthropicEvent("content_block_start", map[string]any{"type": "content_block_start", "index": e.claudeIndex, "content_block": contentBlock}))
	return out
}

func (e *qoderStreamEmitter) claudeCloseOpenBlock() [][]byte {
	if e.claudeOpen == "" {
		return nil
	}
	index := e.claudeIndex
	kind := e.claudeOpen
	e.claudeOpen = ""
	e.claudeIndex++
	return [][]byte{qoderAnthropicEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": index, "content_block": map[string]any{"type": kind}})}
}

func (e *qoderStreamEmitter) claudeToolDone(toolCalls []qoderToolCall) [][]byte {
	out := make([][]byte, 0, len(toolCalls)*3)
	for _, call := range toolCalls {
		index := e.claudeIndex
		e.claudeIndex++
		out = append(out,
			qoderAnthropicEvent("content_block_start", map[string]any{"type": "content_block_start", "index": index, "content_block": map[string]any{"type": "tool_use", "id": valueOrDefault(call.ID, fmt.Sprintf("call_%d", call.Index)), "name": call.Name, "input": map[string]any{}}}),
			qoderAnthropicEvent("content_block_delta", map[string]any{"type": "content_block_delta", "index": index, "delta": map[string]any{"type": "input_json_delta", "partial_json": call.Arguments}}),
			qoderAnthropicEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": index}),
		)
	}
	return out
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
