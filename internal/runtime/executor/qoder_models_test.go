package executor

import "testing"

func TestQoderOpenAIModelsFromCatalog(t *testing.T) {
	raw := []byte(`{
		"chat": [
			{
				"key": "qmodel_latest",
				"display_name": "Qwen3.7-Max",
				"enable": true,
				"is_vl": true,
				"is_reasoning": false,
				"is_default": true,
				"is_free": false,
				"price_factor": 1,
				"max_input_tokens": 128000
			}
		],
		"quest": [
			{"key": "quest_key", "display_name": "Quest Display"}
		]
	}`)

	models, err := qoderOpenAIModelsFromCatalog(raw)
	if err != nil {
		t.Fatalf("qoderOpenAIModelsFromCatalog() error = %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("len(models) = %d, want 1", len(models))
	}
	if got := models[0]["id"]; got != "Qwen3.7-Max" {
		t.Fatalf("model id = %v, want Qwen3.7-Max", got)
	}
	if got := models[0]["owned_by"]; got != "qoder" {
		t.Fatalf("owned_by = %v, want qoder", got)
	}
	if got := models[0]["is_reasoning"]; got != true {
		t.Fatalf("is_reasoning = %v, want true for qmodel_latest", got)
	}
	architecture, ok := models[0]["architecture"].(map[string]any)
	if !ok {
		t.Fatalf("architecture = %#v, want object", models[0]["architecture"])
	}
	if got := architecture["modality"]; got != "text+image->text" {
		t.Fatalf("architecture.modality = %v, want text+image->text", got)
	}
}

func TestQoderModelKeysFromCatalog(t *testing.T) {
	raw := []byte(`{
		"chat": [{"key": "chat_key", "display_name": "Chat Display"}],
		"assistant": [{"key": "assistant_key", "display_name": "Assistant Display"}]
	}`)

	keys := qoderModelKeysFromCatalog(raw)
	if got := keys["Chat Display"]; got != "chat_key" {
		t.Fatalf("Chat Display key = %q, want chat_key", got)
	}
	if got := keys["Assistant Display"]; got != "assistant_key" {
		t.Fatalf("Assistant Display key = %q, want assistant_key", got)
	}
}
