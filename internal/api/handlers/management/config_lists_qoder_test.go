package management

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
)

func TestPutQoderCompatKeepsPATWithoutBaseURL(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	h := &Handler{
		cfg:            &config.Config{},
		configFilePath: writeTestConfigFile(t),
	}

	body := `[{"name":"qoder","base-url":"","api-key-entries":[{"api-key":"pt-123"}],"prefix":"qoder","models":[{"name":"DeepSeek-V4-Pro"}]}]`
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPut, "/v0/management/qoder", strings.NewReader(body))

	h.PutQoderCompat(c)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if got := len(h.cfg.Qoder); got != 1 {
		t.Fatalf("qoder len = %d, want 1", got)
	}
	if got := h.cfg.Qoder[0].BaseURL; got != "" {
		t.Fatalf("base-url = %q, want empty", got)
	}
	if got := h.cfg.Qoder[0].APIKeyEntries[0].APIKey; got != "pt-123" {
		t.Fatalf("api-key = %q, want pt-123", got)
	}
	if got := h.cfg.Qoder[0].Models[0].Name; got != "DeepSeek-V4-Pro" {
		t.Fatalf("model name = %q, want DeepSeek-V4-Pro", got)
	}
}
