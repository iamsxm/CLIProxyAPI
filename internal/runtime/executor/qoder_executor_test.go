package executor

import (
	"encoding/json"
	"testing"

	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestQoderEncodeDecodeRoundTrip(t *testing.T) {
	input := []byte(`{"hello":"world","n":123,"list":["a","b"]}`)
	encoded, err := qoderEncode(input)
	if err != nil {
		t.Fatalf("qoderEncode error: %v", err)
	}
	decoded, err := qoderDecode(encoded)
	if err != nil {
		t.Fatalf("qoderDecode error: %v", err)
	}
	if string(decoded) != string(input) {
		t.Fatalf("round-trip mismatch: got %s want %s", string(decoded), string(input))
	}
}

func TestQoderControlSignature(t *testing.T) {
	got := qoderControlSignature("Tue, 05 May 2026 00:00:00 GMT")
	const want = "57cd3d76f2a94f0306e3026e8c641e38"
	if got != want {
		t.Fatalf("qoderControlSignature = %q, want %q", got, want)
	}
}

func TestQoderPromptExtraction(t *testing.T) {
	openaiReq := []byte(`{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}`)
	gotOpenAI := qoderPromptFromPayload(openaiReq, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai")})
	wantOpenAI := "system:\nsys\n\nuser:\nhello\n\nassistant:\nworld"
	if gotOpenAI != wantOpenAI {
		t.Fatalf("openai prompt = %q, want %q", gotOpenAI, wantOpenAI)
	}

	responsesReq := []byte(`{"instructions":"sys","input":[{"role":"user","content":"hello"},{"type":"text","content":"world"}]}`)
	gotResponses := qoderPromptFromPayload(responsesReq, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")})
	wantResponses := "system:\nsys\n\nuser:\nhello\n\ntext:\nworld"
	if gotResponses != wantResponses {
		t.Fatalf("responses prompt = %q, want %q", gotResponses, wantResponses)
	}

	anthropicReq := []byte(`{"system":"sys","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}`)
	gotClaude := qoderPromptFromPayload(anthropicReq, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude")})
	wantClaude := "system:\nsys\n\nuser:\nhello\n\nassistant:\nworld"
	if gotClaude != wantClaude {
		t.Fatalf("anthropic prompt = %q, want %q", gotClaude, wantClaude)
	}
}

func TestExtractQoderContent(t *testing.T) {
	line := `{"body":"{\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}"}`
	got := extractQoderContent(line)
	if got != "hello" {
		t.Fatalf("extractQoderContent = %q, want %q", got, "hello")
	}
}

func TestQoderNonStreamResponses(t *testing.T) {
	openai := qoderNonStreamResponse("openai", "lite", "hello")
	if got := gjson.GetBytes(openai, "choices.0.message.content").String(); got != "hello" {
		t.Fatalf("openai response content = %q, want %q", got, "hello")
	}

	responses := qoderNonStreamResponse("openai-response", "lite", "hello")
	if got := gjson.GetBytes(responses, "output.0.content.0.text").String(); got != "hello" {
		t.Fatalf("responses output text = %q, want %q", got, "hello")
	}

	claude := qoderNonStreamResponse("claude", "lite", "hello")
	if got := gjson.GetBytes(claude, "content.0.text").String(); got != "hello" {
		t.Fatalf("claude content = %q, want %q", got, "hello")
	}
}

func TestBuildQoderBody(t *testing.T) {
	body, source, err := buildQoderBody("lite", "hello", "personal_standard")
	if err != nil {
		t.Fatalf("buildQoderBody error: %v", err)
	}
	if source == "" {
		t.Fatal("expected model source")
	}
	if got := gjson.GetBytes(body, "stream").Bool(); !got {
		t.Fatal("expected stream=true")
	}
	if got := gjson.GetBytes(body, "model_config.key").String(); got != "lite" {
		t.Fatalf("model_config.key = %q, want %q", got, "lite")
	}
	if got := gjson.GetBytes(body, "chat_context.text.text").String(); got != "hello" {
		t.Fatalf("chat_context.text.text = %q, want %q", got, "hello")
	}
	if got := gjson.GetBytes(body, "messages.#").Int(); got == 0 {
		t.Fatal("expected messages to remain populated")
	}
}

func TestQoderUserMessageJSON(t *testing.T) {
	raw, err := qoderUserMessage("hello")
	if err != nil {
		t.Fatalf("qoderUserMessage error: %v", err)
	}
	var msg map[string]any
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if msg["role"] != "user" {
		t.Fatalf("role = %v, want user", msg["role"])
	}
}
