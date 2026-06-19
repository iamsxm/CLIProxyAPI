package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

const (
	qoderModelListURL     = "https://api1.qoder.sh/algo/api/v2/model/list?Encode=1"
	qoderModelListPath    = "/api/v2/model/list"
	qoderModelKeyCacheTTL = 5 * time.Minute
)

var qoderStaticModelKeyMap = map[string]string{
	"Qwen3.7-Max": "qmodel_latest",
}

var qoderModelKeyCatalogSections = []string{
	"chat",
	"quest",
	"nap",
	"qwork",
	"experts",
	"qwake",
	"assistant",
	"inline",
}

// FetchOpenAIModels fetches Qoder's native model catalog and formats the chat
// models as an OpenAI /v1/models-compatible list.
func (e *QoderExecutor) FetchOpenAIModels(ctx context.Context, auth *cliproxyauth.Auth) ([]map[string]any, error) {
	raw, err := e.fetchQoderModelCatalog(ctx, auth)
	if err != nil {
		return nil, err
	}
	e.updateQoderModelKeyCache(raw)
	models, err := qoderOpenAIModelsFromCatalog(raw)
	if err != nil {
		return nil, err
	}
	if len(models) == 0 {
		return qoderStaticOpenAIModels(), nil
	}
	return models, nil
}

// resolveQoderModel mirrors qoder2api's dynamic display_name -> key mapping so
// models discovered from /v1/models can be used directly in chat requests.
func (e *QoderExecutor) resolveQoderModel(ctx context.Context, auth *cliproxyauth.Auth, model string) string {
	base := qoderNormalizeModel(model)
	if base == "" || base == qoderDefaultModel {
		return base
	}
	if mapped := e.cachedQoderModelKey(base); mapped != "" {
		return mapped
	}
	raw, err := e.fetchQoderModelCatalog(ctx, auth)
	if err != nil {
		return base
	}
	keys := e.updateQoderModelKeyCache(raw)
	if mapped := keys[base]; mapped != "" {
		return mapped
	}
	return base
}

func (e *QoderExecutor) cachedQoderModelKey(model string) string {
	if e == nil {
		return ""
	}
	e.modelMu.Lock()
	defer e.modelMu.Unlock()
	if e.modelKeyMap == nil || time.Since(e.modelKeysAt) >= qoderModelKeyCacheTTL {
		return ""
	}
	return e.modelKeyMap[model]
}

// updateQoderModelKeyCache rebuilds the display-name mapping from every Qoder
// catalog section that may contain callable or assistant models.
func (e *QoderExecutor) updateQoderModelKeyCache(raw []byte) map[string]string {
	keys := qoderModelKeysFromCatalog(raw)
	if e != nil && len(keys) > 0 {
		e.modelMu.Lock()
		e.modelKeyMap = keys
		e.modelKeysAt = time.Now()
		e.modelMu.Unlock()
	}
	return keys
}

// fetchQoderModelCatalog signs a GET request with an empty encoded body, which
// matches qoder2api's BearerApiClient.callGet behavior for catalog discovery.
func (e *QoderExecutor) fetchQoderModelCatalog(ctx context.Context, auth *cliproxyauth.Auth) ([]byte, error) {
	sess, err := e.getOrCreateSession(ctx, auth)
	if err != nil {
		return nil, err
	}
	bearer, cosyDate, err := buildQoderBearer(sess, "", qoderModelListPath)
	if err != nil {
		return nil, err
	}
	reqCtx := ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodGet, qoderModelListURL, nil)
	if err != nil {
		return nil, err
	}
	applyQoderJSONHeaders(httpReq, sess, bearer, cosyDate)

	httpClient := helps.NewProxyAwareHTTPClient(reqCtx, e.cfg, auth, qoderRequestTimeout)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		return nil, statusErr{code: httpResp.StatusCode, msg: string(body)}
	}
	return body, nil
}

// applyQoderJSONHeaders reproduces the signed JSON headers used by qoder2api's
// generic bearer client for non-streaming GET/POST calls.
func applyQoderJSONHeaders(req *http.Request, sess *qoderSession, bearer, cosyDate string) {
	req.Header.Set("cosy-data-policy", "AGREE")
	req.Header.Set("content-type", "application/json")
	req.Header.Set("cosy-machinetype", sess.machineType)
	req.Header.Set("cosy-clienttype", qoderClientType)
	req.Header.Set("cosy-date", cosyDate)
	req.Header.Set("cosy-user", sess.identity.UID)
	req.Header.Set("cosy-key", sess.cosyKey)
	req.Header.Set("accept", "application/json")
	req.Header.Set("cosy-clientip", qoderClientIP)
	req.Header.Set("authorization", bearer)
	req.Header.Set("accept-encoding", "identity")
	req.Header.Set("cosy-version", qoderVersion)
	req.Header.Set("cosy-machineid", sess.machineID)
	req.Header.Set("cosy-machinetoken", sess.machineToken)
	req.Header.Set("login-version", "v2")
	req.Header.Set("user-agent", qoderUserAgent)
}

// qoderOpenAIModelsFromCatalog extracts the catalog's chat models and keeps the
// native Qoder capability flags that qoder2api exposes from /v1/models.
func qoderOpenAIModelsFromCatalog(raw []byte) ([]map[string]any, error) {
	var catalog map[string]json.RawMessage
	if err := json.Unmarshal(raw, &catalog); err != nil {
		return nil, fmt.Errorf("parse qoder model catalog: %w", err)
	}
	chatRaw := catalog["chat"]
	if len(chatRaw) == 0 || string(chatRaw) == "null" {
		return nil, nil
	}
	var chat []map[string]any
	if err := json.Unmarshal(chatRaw, &chat); err != nil {
		return nil, fmt.Errorf("parse qoder chat models: %w", err)
	}
	out := make([]map[string]any, 0, len(chat))
	for _, model := range chat {
		entry := qoderOpenAIModelEntry(model)
		if strings.TrimSpace(stringValueAny(entry["id"])) == "" {
			continue
		}
		out = append(out, entry)
	}
	return out, nil
}

func qoderModelKeysFromCatalog(raw []byte) map[string]string {
	var catalog map[string]json.RawMessage
	if err := json.Unmarshal(raw, &catalog); err != nil {
		return nil
	}
	out := make(map[string]string)
	for _, section := range qoderModelKeyCatalogSections {
		var models []map[string]any
		if rawSection := catalog[section]; len(rawSection) == 0 || string(rawSection) == "null" {
			continue
		} else if err := json.Unmarshal(rawSection, &models); err != nil {
			continue
		}
		for _, model := range models {
			displayName := stringValueAny(model["display_name"])
			key := stringValueAny(model["key"])
			if displayName != "" && key != "" {
				out[displayName] = key
			}
		}
	}
	return out
}

func qoderOpenAIModelEntry(model map[string]any) map[string]any {
	id := stringValueAny(model["display_name"])
	if id == "" {
		id = stringValueAny(model["key"])
	}
	key := stringValueAny(model["key"])
	isVL := boolValueAny(model["is_vl"])
	isReasoning := key == "qmodel_latest" || boolValueAny(model["is_reasoning"])
	entry := map[string]any{
		"id":               id,
		"display_name":     stringValueAny(model["display_name"]),
		"object":           "model",
		"created":          0,
		"owned_by":         "qoder",
		"enable":           boolValueAny(model["enable"]),
		"is_reasoning":     isReasoning,
		"is_vl":            isVL,
		"is_default":       boolValueAny(model["is_default"]),
		"is_new":           boolValueAny(model["is_new"]),
		"is_free":          boolValueAny(model["is_free"]),
		"is_editable":      boolValueAny(model["is_editable"]),
		"price_factor":     model["price_factor"],
		"max_input_tokens": model["max_input_tokens"],
		"architecture": map[string]any{
			"modality":          qoderModelModality(isVL),
			"input_modalities":  qoderInputModalities(isVL),
			"output_modalities": []string{"text"},
		},
	}
	if v, ok := model["original_price_factor"]; ok {
		entry["original_price_factor"] = v
	}
	for _, key := range []string{"context_config", "thinking_config", "minimal_version"} {
		if v, ok := model[key]; ok {
			entry[key] = v
		}
	}
	return entry
}

func qoderStaticOpenAIModels() []map[string]any {
	out := make([]map[string]any, 0, len(qoderStaticModelKeyMap))
	for displayName := range qoderStaticModelKeyMap {
		out = append(out, map[string]any{
			"id":       displayName,
			"object":   "model",
			"created":  0,
			"owned_by": "qoder",
		})
	}
	return out
}

func qoderModelModality(isVL bool) string {
	if isVL {
		return "text+image->text"
	}
	return "text->text"
}

func qoderInputModalities(isVL bool) []string {
	if isVL {
		return []string{"text", "image"}
	}
	return []string{"text"}
}

func boolValueAny(value any) bool {
	v, _ := value.(bool)
	return v
}

func stringValueAny(value any) string {
	v, _ := value.(string)
	return strings.TrimSpace(v)
}
