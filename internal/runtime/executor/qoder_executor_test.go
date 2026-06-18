package executor

import (
	"encoding/json"
	"net/http"
	"testing"

	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
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
	openai := qoderNonStreamResponse("openai", "lite", qoderTextCompletion("hello"))
	if got := gjson.GetBytes(openai, "choices.0.message.content").String(); got != "hello" {
		t.Fatalf("openai response content = %q, want %q", got, "hello")
	}

	responses := qoderNonStreamResponse("openai-response", "lite", qoderTextCompletion("hello"))
	if got := gjson.GetBytes(responses, "output.0.content.0.text").String(); got != "hello" {
		t.Fatalf("responses output text = %q, want %q", got, "hello")
	}

	claude := qoderNonStreamResponse("claude", "lite", qoderTextCompletion("hello"))
	if got := gjson.GetBytes(claude, "content.0.text").String(); got != "hello" {
		t.Fatalf("claude content = %q, want %q", got, "hello")
	}
}

func TestExtractQoderDeltaStructured(t *testing.T) {
	line := `{"body":"{\"choices\":[{\"delta\":{\"content\":\"hello\",\"reasoning_content\":\"thinking\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}"}`
	got, err := extractQoderDelta(line)
	if err != nil {
		t.Fatalf("extractQoderDelta error: %v", err)
	}
	if got.Content != "hello" {
		t.Fatalf("content = %q, want hello", got.Content)
	}
	if got.Reasoning != "thinking" {
		t.Fatalf("reasoning = %q, want thinking", got.Reasoning)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].ID != "call_1" || got.ToolCalls[0].Name != "read" || got.ToolCalls[0].Arguments != `{"path":` {
		t.Fatalf("tool call delta mismatch: %+v", got.ToolCalls)
	}
}

func TestQoderNonStreamStructuredOutput(t *testing.T) {
	completion := newQoderCompletion()
	completion.Add(qoderDelta{
		Content:   "answer",
		Reasoning: "plan",
		ToolCalls: []qoderToolCallDelta{{
			Index:     0,
			ID:        "call_1",
			Type:      "function",
			Name:      "read",
			Arguments: `{"path":"README.md"}`,
		}},
	})

	openai := qoderNonStreamResponse("openai", "lite", completion)
	if got := gjson.GetBytes(openai, "choices.0.message.reasoning_content").String(); got != "plan" {
		t.Fatalf("openai reasoning_content = %q, want plan; body=%s", got, string(openai))
	}
	if got := gjson.GetBytes(openai, "choices.0.message.tool_calls.0.function.name").String(); got != "read" {
		t.Fatalf("openai tool name = %q, want read; body=%s", got, string(openai))
	}
	if got := gjson.GetBytes(openai, "choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("openai finish_reason = %q, want tool_calls", got)
	}

	responses := qoderNonStreamResponse("openai-response", "lite", completion)
	if got := gjson.GetBytes(responses, `output.#(type=="reasoning").summary.0.text`).String(); got != "plan" {
		t.Fatalf("responses reasoning summary = %q, want plan; body=%s", got, string(responses))
	}
	if got := gjson.GetBytes(responses, `output.#(type=="function_call").name`).String(); got != "read" {
		t.Fatalf("responses function_call name = %q, want read; body=%s", got, string(responses))
	}

	claude := qoderNonStreamResponse("claude", "lite", completion)
	if got := gjson.GetBytes(claude, `content.#(type=="thinking").thinking`).String(); got != "plan" {
		t.Fatalf("claude thinking = %q, want plan; body=%s", got, string(claude))
	}
	if got := gjson.GetBytes(claude, `content.#(type=="tool_use").name`).String(); got != "read" {
		t.Fatalf("claude tool_use name = %q, want read; body=%s", got, string(claude))
	}
	if got := gjson.GetBytes(claude, "stop_reason").String(); got != "tool_use" {
		t.Fatalf("claude stop_reason = %q, want tool_use", got)
	}
}

func TestQoderNonStreamParsesToolCallsTextFallback(t *testing.T) {
	completion := qoderTextCompletion(`Tool calls: [{"id":"call_1","type":"function","function":{"name":"read","arguments":{"path":"README.md"}}}]`)
	openai := qoderNonStreamResponse("openai", "lite", completion)
	if got := gjson.GetBytes(openai, "choices.0.message.content").String(); got != "" {
		t.Fatalf("openai content = %q, want empty after tool-call fallback; body=%s", got, string(openai))
	}
	if got := gjson.GetBytes(openai, "choices.0.message.tool_calls.0.function.arguments").String(); got != `{"path":"README.md"}` {
		t.Fatalf("openai tool args = %q, want JSON object string; body=%s", got, string(openai))
	}
	if got := gjson.GetBytes(openai, "choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", got)
	}
}

func TestQoderOpenAIStreamEmitsContentAndToolCallsTogether(t *testing.T) {
	emitter := newQoderStreamEmitter("openai", "lite")
	chunks := emitter.delta(qoderDelta{
		Content: "answer",
		ToolCalls: []qoderToolCallDelta{{
			Index:     0,
			ID:        "call_1",
			Type:      "function",
			Name:      "read",
			Arguments: `{"path":"README.md"}`,
		}},
	})
	if len(chunks) != 1 {
		t.Fatalf("chunks len = %d, want 1", len(chunks))
	}
	if got := gjson.GetBytes(chunks[0], "choices.0.delta.content").String(); got != "answer" {
		t.Fatalf("stream content = %q, want answer; body=%s", got, string(chunks[0]))
	}
	if got := gjson.GetBytes(chunks[0], "choices.0.delta.tool_calls.0.function.name").String(); got != "read" {
		t.Fatalf("stream tool name = %q, want read; body=%s", got, string(chunks[0]))
	}
	done := emitter.done()
	if got := gjson.GetBytes(done[0], "choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("done finish_reason = %q, want tool_calls; body=%s", got, string(done[0]))
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

func TestBuildQoderBodyFromPayloadPreservesMessagesToolsAndImages(t *testing.T) {
	payload := []byte(`{
	  "messages": [
	    {"role":"system","content":"custom system"},
	    {"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]},
	    {"role":"assistant","content":"calling","tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{}"}}]},
	    {"role":"tool","tool_call_id":"call_1","name":"ls","content":"ok"}
	  ],
	  "tools": [{"type":"function","function":{"name":"ls","parameters":{"type":"object"}}}]
	}`)
	body, _, err := buildQoderBodyFromPayload("qmodel_latest", "look", "personal_standard", payload)
	if err != nil {
		t.Fatalf("buildQoderBodyFromPayload error: %v", err)
	}
	if got := gjson.GetBytes(body, "messages.0.role").String(); got != "system" {
		t.Fatalf("messages.0.role = %q, want system; body=%s", got, string(body))
	}
	if got := gjson.GetBytes(body, "messages.0.content").String(); got != "custom system" {
		t.Fatalf("custom system not preserved, got %q", got)
	}
	if got := gjson.GetBytes(body, "messages.1.contents.1.image_url.url").String(); got != "data:image/png;base64,abc" {
		t.Fatalf("image_url not preserved, got %q; body=%s", got, string(body))
	}
	if got := gjson.GetBytes(body, "messages.2.tool_calls.0.function.name").String(); got != "ls" {
		t.Fatalf("assistant tool call name = %q, want ls; body=%s", got, string(body))
	}
	if got := gjson.GetBytes(body, "messages.3.tool_call_id").String(); got != "call_1" {
		t.Fatalf("tool_call_id = %q, want call_1; body=%s", got, string(body))
	}
	if got := gjson.GetBytes(body, "tools.0.function.name").String(); got != "ls" {
		t.Fatalf("tools not copied, got %q; body=%s", got, string(body))
	}
}

func TestBuildQoderBodyFromPayloadParsesAssistantToolCallsText(t *testing.T) {
	payload := []byte(`{
	  "messages": [
	    {"role":"assistant","content":"Tool calls: [{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"README.md\"}}}]"}
	  ],
	  "tools": [{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}]
	}`)
	body, _, err := buildQoderBodyFromPayload("qmodel_latest", "read", "personal_standard", payload)
	if err != nil {
		t.Fatalf("buildQoderBodyFromPayload error: %v", err)
	}
	if got := gjson.GetBytes(body, "messages.1.tool_calls.0.function.name").String(); got != "read" {
		t.Fatalf("tool call text fallback name = %q, want read; body=%s", got, string(body))
	}
	if got := gjson.GetBytes(body, "messages.1.content").String(); got != "" {
		t.Fatalf("assistant fallback content = %q, want empty; body=%s", got, string(body))
	}
}

func TestQoderPromptForRequestUsesLatestUserMessage(t *testing.T) {
	req := cliproxyexecutor.Request{Payload: []byte(`{"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"ok"},{"role":"user","content":"second"}]}`)}
	got := qoderPromptForRequest(req, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI})
	if got != "second" {
		t.Fatalf("qoderPromptForRequest = %q, want second", got)
	}
}

func TestQoderRequestToFormatUsesOpenAIChat(t *testing.T) {
	exec := NewQoderExecutor(nil)
	if got := exec.RequestToFormat(cliproxyexecutor.Request{}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude}); got != sdktranslator.FormatOpenAI {
		t.Fatalf("RequestToFormat = %q, want %q", got, sdktranslator.FormatOpenAI)
	}
}

func TestQoderModelAlias(t *testing.T) {
	if got := qoderNormalizeModel("Qwen3.7-Max"); got != "qmodel_latest" {
		t.Fatalf("qoderNormalizeModel = %q, want qmodel_latest", got)
	}
}

func TestExtractQoderDeltaQuotaError(t *testing.T) {
	_, err := extractQoderDelta(`{"body":"{\"code\":\"429\",\"message\":\"quota exhausted\"}"}`)
	if err == nil {
		t.Fatal("expected quota error")
	}
	statusErr, ok := err.(cliproxyexecutor.StatusError)
	if !ok {
		t.Fatalf("error type = %T, want StatusError", err)
	}
	if statusErr.StatusCode() != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", statusErr.StatusCode(), http.StatusTooManyRequests)
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
