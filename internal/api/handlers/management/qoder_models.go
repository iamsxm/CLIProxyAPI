package management

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	runtimeexecutor "github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

type qoderModelsRequest struct {
	APIKey         string `json:"api-key"`
	APIKeySnake    string `json:"api_key"`
	APIKeyCamel    string `json:"apiKey"`
	AuthIndexSnake string `json:"auth-index"`
	AuthIndexAlt   string `json:"auth_index"`
	AuthIndexCamel string `json:"authIndex"`
}

// GetQoderModels returns Qoder's native model catalog in the same shape as
// qoder2api's /v1/models endpoint.
func (h *Handler) GetQoderModels(c *gin.Context) {
	var body qoderModelsRequest
	if c.Request != nil && c.Request.Method != http.MethodGet && c.Request.Body != nil {
		if err := c.ShouldBindJSON(&body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
			return
		}
	}

	apiKey := firstNonEmptyRaw(
		body.APIKey,
		body.APIKeySnake,
		body.APIKeyCamel,
	)
	authIndex := firstNonEmptyRaw(
		body.AuthIndexSnake,
		body.AuthIndexAlt,
		body.AuthIndexCamel,
		c.Query("auth-index"),
		c.Query("auth_index"),
		c.Query("authIndex"),
	)

	auth := h.qoderModelDiscoveryAuth(authIndex, apiKey)
	if auth == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing qoder api key"})
		return
	}

	models, err := h.qoderModelExecutor().FetchOpenAIModels(c.Request.Context(), auth)
	if err != nil {
		statusCode := http.StatusBadGateway
		if withStatus, ok := err.(interface{ StatusCode() int }); ok {
			if code := withStatus.StatusCode(); code >= http.StatusBadRequest {
				statusCode = code
			}
		}
		c.JSON(statusCode, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
}

func (h *Handler) qoderModelExecutor() *runtimeexecutor.QoderExecutor {
	if h == nil {
		return runtimeexecutor.NewQoderExecutor(nil)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.qoderExecutor == nil {
		h.qoderExecutor = runtimeexecutor.NewQoderExecutor(h.cfg)
	}
	return h.qoderExecutor
}

// qoderModelDiscoveryAuth builds a disposable auth record for catalog discovery.
// It avoids mutating stored auth entries while still preserving per-key proxy settings.
func (h *Handler) qoderModelDiscoveryAuth(authIndex, apiKey string) *coreauth.Auth {
	apiKey = strings.TrimSpace(apiKey)
	authIndex = strings.TrimSpace(authIndex)
	if authIndex != "" {
		if existing := h.authByIndex(authIndex); existing != nil {
			auth := existing.Clone()
			if apiKey != "" {
				ensureAuthAttributes(auth)
				auth.Attributes["api_key"] = apiKey
			}
			ensureQoderAuthMetadata(auth)
			return auth
		}
	}
	if apiKey != "" {
		return newQoderDiscoveryAuth(apiKey, h.qoderProxyForAPIKey(apiKey))
	}
	if h == nil || h.cfg == nil {
		return nil
	}
	for i := range h.cfg.Qoder {
		entry := h.cfg.Qoder[i]
		if entry.Disabled {
			continue
		}
		if key, proxyURL := firstQoderAPIKeyEntry(entry); key != "" {
			return newQoderDiscoveryAuth(key, proxyURL)
		}
	}
	return nil
}

func newQoderDiscoveryAuth(apiKey, proxyURL string) *coreauth.Auth {
	auth := &coreauth.Auth{
		ID:       "management-qoder-models",
		Provider: "qoder",
		ProxyURL: strings.TrimSpace(proxyURL),
		Attributes: map[string]string{
			"api_key":      strings.TrimSpace(apiKey),
			"compat_name":  "qoder",
			"provider_key": "qoder",
		},
	}
	auth.EnsureIndex()
	return auth
}

func ensureAuthAttributes(auth *coreauth.Auth) {
	if auth != nil && auth.Attributes == nil {
		auth.Attributes = map[string]string{}
	}
}

func ensureQoderAuthMetadata(auth *coreauth.Auth) {
	if auth == nil {
		return
	}
	if strings.TrimSpace(auth.Provider) == "" {
		auth.Provider = "qoder"
	}
	ensureAuthAttributes(auth)
	if strings.TrimSpace(auth.Attributes["compat_name"]) == "" {
		auth.Attributes["compat_name"] = "qoder"
	}
	if strings.TrimSpace(auth.Attributes["provider_key"]) == "" {
		auth.Attributes["provider_key"] = "qoder"
	}
}

func firstQoderAPIKeyEntry(entry config.OpenAICompatibility) (string, string) {
	for i := range entry.APIKeyEntries {
		apiKey := strings.TrimSpace(entry.APIKeyEntries[i].APIKey)
		if apiKey == "" {
			continue
		}
		return apiKey, strings.TrimSpace(entry.APIKeyEntries[i].ProxyURL)
	}
	return "", ""
}

func (h *Handler) qoderProxyForAPIKey(apiKey string) string {
	if h == nil || h.cfg == nil {
		return ""
	}
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return ""
	}
	for i := range h.cfg.Qoder {
		entry := h.cfg.Qoder[i]
		if entry.Disabled {
			continue
		}
		for j := range entry.APIKeyEntries {
			keyEntry := entry.APIKeyEntries[j]
			if strings.EqualFold(strings.TrimSpace(keyEntry.APIKey), apiKey) {
				return strings.TrimSpace(keyEntry.ProxyURL)
			}
		}
	}
	return ""
}

func firstNonEmptyRaw(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
