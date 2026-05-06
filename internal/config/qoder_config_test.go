package config

import "testing"

func TestSanitizeQoderCompatibilityKeepsPATWithoutBaseURL(t *testing.T) {
	cfg := &Config{
		Qoder: []OpenAICompatibility{
			{
				Name:    " ",
				BaseURL: "",
				APIKeyEntries: []OpenAICompatibilityAPIKey{
					{APIKey: " pt-123 "},
				},
			},
			{
				Name:    "empty",
				BaseURL: "",
			},
		},
	}

	cfg.SanitizeQoderCompatibility()

	if len(cfg.Qoder) != 1 {
		t.Fatalf("len(Qoder) = %d, want 1", len(cfg.Qoder))
	}
	if cfg.Qoder[0].Name != "qoder" {
		t.Fatalf("Qoder[0].Name = %q, want qoder", cfg.Qoder[0].Name)
	}
	if cfg.Qoder[0].BaseURL != "" {
		t.Fatalf("Qoder[0].BaseURL = %q, want empty", cfg.Qoder[0].BaseURL)
	}
	if got := cfg.Qoder[0].APIKeyEntries[0].APIKey; got != "pt-123" {
		t.Fatalf("APIKey = %q, want pt-123", got)
	}
}
