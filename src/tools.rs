//! Tool call adaptation: injects tool definitions into system prompt for models
//! without native function calling, and parses `<tool_call>` tags from output.

use serde_json::{json, Value};
use tracing::info;

pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: Option<String>,
}

/// Events emitted by the streaming tool call parser.
pub enum ToolStreamEvent {
    Text(String),
    ToolCallComplete(ToolCall),
}

/// Transform request: inject tools into system prompt, remove `tools` field.
/// Returns true if the request was modified.
pub fn transform_request(data: &mut Value) -> bool {
    let tools = match data.get("tools").and_then(|t| t.as_array()) {
        Some(t) if !t.is_empty() => t.clone(),
        _ => return false,
    };

    let tool_choice = data.get("tool_choice").cloned();
    let tools_prompt = build_tools_prompt(&tools, &tool_choice);

    if let Some(messages) = data.get_mut("messages").and_then(|m| m.as_array_mut()) {
        transform_messages(messages);

        let has_system = messages.first()
            .map_or(false, |m| m.get("role").and_then(|r| r.as_str()) == Some("system"));

        if has_system {
            if let Some(sys) = messages.first_mut() {
                if let Some(content) = sys.get("content").and_then(|c| c.as_str()) {
                    sys["content"] = Value::String(format!("{}\n\n{}", content, tools_prompt));
                }
            }
        } else {
            messages.insert(0, json!({"role": "system", "content": tools_prompt}));
        }
    }

    if let Some(obj) = data.as_object_mut() {
        obj.remove("tools");
        obj.remove("tool_choice");
    }

    info!("Tool adaptation: injected {} tools into system prompt", tools.len());
    true
}

fn build_tools_prompt(tools: &[Value], tool_choice: &Option<Value>) -> String {
    let mut prompt = String::from(
        "# Available Tools\n\n\
         You have access to the following tools. To use a tool, output a tool call in this exact format:\n\n\
         <tool_call>\n\
         {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n\
         </tool_call>\n\n\
         IMPORTANT RULES for tool calls:\n\
         1. Each tool call must be wrapped in its own <tool_call></tool_call> tags.\n\
         2. The JSON inside must be valid. For string values containing newlines, use \\n. For quotes, use \\\".\n\
         3. For file write operations: put the COMPLETE file content in the \"content\" argument as a single JSON string. \
            Every newline in the file must be written as \\n in the JSON string.\n\
         4. When you want to call a tool, output ONLY the tool call(s) with no other text.\n\
         5. When you don't need to use any tool, respond normally without <tool_call> tags.\n\n\
         Example of writing a file:\n\
         <tool_call>\n\
         {\"name\": \"Write\", \"arguments\": {\"file_path\": \"/path/to/file.py\", \"content\": \"#!/usr/bin/env python3\\nimport os\\n\\ndef main():\\n    print(\\\"hello\\\")\\n\"}}\n\
         </tool_call>\n\n\
         ## Tool Definitions\n\n"
    );

    for tool in tools {
        let (name, desc, params) = if let Some(func) = tool.get("function") {
            (
                func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown"),
                func.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                func.get("parameters").map(|p| serde_json::to_string_pretty(p).unwrap_or_default()).unwrap_or_default(),
            )
        } else {
            (
                tool.get("name").and_then(|n| n.as_str()).unwrap_or("unknown"),
                tool.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                tool.get("input_schema").or_else(|| tool.get("parameters"))
                    .map(|p| serde_json::to_string_pretty(p).unwrap_or_default()).unwrap_or_default(),
            )
        };
        prompt.push_str(&format!("### {}\n{}\nParameters:\n```json\n{}\n```\n\n", name, desc, params));
    }

    if let Some(choice) = tool_choice {
        match choice {
            Value::String(s) if s == "required" || s == "any" => {
                prompt.push_str("IMPORTANT: You MUST use at least one tool in your response.\n\n");
            }
            Value::String(s) if s == "none" => {
                prompt.push_str("NOTE: Do not use any tools. Respond with text only.\n\n");
            }
            Value::Object(obj) => {
                let func_name = obj.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str())
                    .or_else(|| obj.get("name").and_then(|n| n.as_str()));
                if let Some(name) = func_name {
                    prompt.push_str(&format!("IMPORTANT: You MUST use the \"{}\" tool in your response.\n\n", name));
                }
            }
            _ => {}
        }
    }

    prompt
}

fn transform_messages(messages: &mut Vec<Value>) {
    for msg in messages.iter_mut() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
        match role.as_str() {
            "tool" => {
                let tool_call_id = msg.get("tool_call_id").and_then(|t| t.as_str()).unwrap_or("").to_string();
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                msg["role"] = Value::String("user".to_string());
                msg["content"] = Value::String(format!("[Tool Result (call_id: {})]:\n{}", tool_call_id, content));
                if let Some(obj) = msg.as_object_mut() { obj.remove("tool_call_id"); }
            }
            "assistant" => {
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                    let mut text_parts = Vec::new();
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() { text_parts.push(content.to_string()); }
                    }
                    for tc in tool_calls {
                        if let Some(func) = tc.get("function") {
                            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let args = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                            let args_json = serde_json::from_str::<Value>(args)
                                .and_then(|v| serde_json::to_string(&v))
                                .unwrap_or_else(|_| args.to_string());
                            text_parts.push(format!(
                                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}, \"call_id\": \"{}\"}}\n</tool_call>",
                                name, args_json, id
                            ));
                        }
                    }
                    msg["content"] = Value::String(text_parts.join("\n"));
                    if let Some(obj) = msg.as_object_mut() { obj.remove("tool_calls"); }
                }
            }
            _ => {}
        }
    }
}

/// Convert Anthropic request to OpenAI format.
pub fn anthropic_request_to_openai(req: &Value) -> Value {
    let mut messages = Vec::new();

    // System prompt
    if let Some(system) = req.get("system") {
        let text = extract_anthropic_text(system);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    // Messages
    if let Some(msgs) = req.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").unwrap_or(&Value::Null);

            match content {
                Value::String(s) => {
                    messages.push(json!({"role": role, "content": s}));
                }
                Value::Array(blocks) => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    let mut tool_results = Vec::new();

                    for block in blocks {
                        let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match btype {
                            "text" => {
                                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                    text_parts.push(t.to_string());
                                }
                            }
                            "tool_use" => {
                                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                                let empty_obj = json!({});
                                let input = block.get("input").unwrap_or(&empty_obj);
                                let args_str = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                                tool_calls.push(json!({
                                    "id": id, "type": "function",
                                    "function": {"name": name, "arguments": args_str}
                                }));
                            }
                            "tool_result" => {
                                let tool_use_id = block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or("");
                                let result_content = block.get("content").map(|c| extract_anthropic_text(c)).unwrap_or_default();
                                let is_error = block.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false);
                                let prefix = if is_error { "[ERROR] " } else { "" };
                                tool_results.push((tool_use_id.to_string(), format!("{}{}", prefix, result_content)));
                            }
                            "thinking" => {} // skip
                            _ => {}
                        }
                    }

                    if role == "assistant" {
                        let mut msg = json!({"role": "assistant"});
                        let text = text_parts.join("");
                        if !text.is_empty() { msg["content"] = Value::String(text); }
                        if !tool_calls.is_empty() {
                            msg["tool_calls"] = Value::Array(tool_calls);
                            if msg.get("content").is_none() { msg["content"] = Value::Null; }
                        } else if msg.get("content").is_none() {
                            msg["content"] = Value::String(String::new());
                        }
                        messages.push(msg);
                    } else {
                        if !tool_results.is_empty() {
                            for (tool_use_id, result) in &tool_results {
                                messages.push(json!({"role": "tool", "tool_call_id": tool_use_id, "content": result}));
                            }
                            let text = text_parts.join("");
                            if !text.is_empty() { messages.push(json!({"role": "user", "content": text})); }
                        } else {
                            messages.push(json!({"role": role, "content": text_parts.join("")}));
                        }
                    }
                }
                _ => {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
    }

    let mut result = json!({
        "model": req.get("model").and_then(|m| m.as_str()).unwrap_or(""),
        "messages": messages,
        "max_tokens": req.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(4096),
        "stream": req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
    });

    if let Some(temp) = req.get("temperature") { result["temperature"] = temp.clone(); }
    if let Some(top_p) = req.get("top_p") { result["top_p"] = top_p.clone(); }
    if let Some(stop) = req.get("stop_sequences") { result["stop"] = stop.clone(); }

    // Pass through sampling params for local model control
    for key in &["repetition_penalty", "repetition_context_size", "top_k", "min_p", "frequency_penalty", "presence_penalty"] {
        if let Some(v) = req.get(*key) { result[*key] = v.clone(); }
    }

    // Convert Anthropic tools → OpenAI format
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let openai_tools: Vec<Value> = tools.iter().map(|t| {
                if t.get("type").and_then(|ty| ty.as_str()) == Some("function") {
                    t.clone()
                } else {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                            "description": t.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                            "parameters": t.get("input_schema").unwrap_or(&json!({"type": "object"}))
                        }
                    })
                }
            }).collect();
            result["tools"] = Value::Array(openai_tools);
        }
    }

    if let Some(choice) = req.get("tool_choice") {
        result["tool_choice"] = match choice {
            Value::Object(obj) => {
                match obj.get("type").and_then(|t| t.as_str()).unwrap_or("auto") {
                    "auto" => json!("auto"),
                    "any" => json!("required"),
                    "tool" => obj.get("name").and_then(|n| n.as_str())
                        .map(|name| json!({"type": "function", "function": {"name": name}}))
                        .unwrap_or(json!("auto")),
                    _ => json!("auto"),
                }
            }
            _ => choice.clone(),
        };
    }

    if result.get("stream").and_then(|s| s.as_bool()).unwrap_or(false) {
        result["stream_options"] = json!({"include_usage": true});
    }

    result
}

fn extract_anthropic_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks.iter().filter_map(|b| {
            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
            } else { None }
        }).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// Parse model output for `<tool_call>` tags and transform response.
pub fn transform_response(resp_body: bytes::Bytes) -> bytes::Bytes {
    let mut json: Value = match serde_json::from_slice(&resp_body) {
        Ok(v) => v,
        Err(_) => return resp_body,
    };

    let choices = match json.get_mut("choices").and_then(|c| c.as_array_mut()) {
        Some(c) => c,
        None => return resp_body,
    };

    let mut modified = false;
    for choice in choices.iter_mut() {
        let msg = match choice.get_mut("message") { Some(m) => m, None => continue };
        let content = match msg.get("content").and_then(|c| c.as_str()) { Some(c) => c.to_string(), None => continue };

        let tool_calls = parse_tool_calls(&content);
        if tool_calls.is_empty() { continue; }

        let tc_array: Vec<Value> = tool_calls.iter().enumerate().map(|(i, tc)| {
            json!({
                "id": tc.id.as_deref().unwrap_or(&format!("call_{}", i)),
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments}
            })
        }).collect();

        let remaining_text = strip_tool_calls(&content);
        msg["tool_calls"] = Value::Array(tc_array);
        if remaining_text.trim().is_empty() { msg["content"] = Value::Null; }
        else { msg["content"] = Value::String(remaining_text); }
        choice["finish_reason"] = Value::String("tool_calls".to_string());
        modified = true;
    }

    if modified { serde_json::to_vec(&json).map(bytes::Bytes::from).unwrap_or(resp_body) }
    else { resp_body }
}

// ─── Streaming tool call parser ──────────────────────────────────────────────

pub struct ToolCallStreamParser {
    buf: String,
    in_tag: bool,
    tag_buf: String,
    tool_index: usize,
}

impl ToolCallStreamParser {
    pub fn new() -> Self {
        Self { buf: String::new(), in_tag: false, tag_buf: String::new(), tool_index: 0 }
    }

    pub fn feed(&mut self, text: &str) -> Vec<ToolStreamEvent> {
        self.buf.push_str(text);
        let mut events = Vec::new();

        loop {
            if self.in_tag {
                if let Some(end_pos) = self.buf.find("</tool_call>") {
                    self.tag_buf.push_str(&self.buf[..end_pos]);
                    self.buf = self.buf[end_pos + "</tool_call>".len()..].to_string();
                    if let Some(parsed) = fix_json(self.tag_buf.trim()) {
                        let name = parsed.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let arguments = parsed.get("arguments").map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string())).unwrap_or_else(|| "{}".to_string());
                        let id = parsed.get("call_id").and_then(|i| i.as_str()).map(|s| s.to_string())
                            .or_else(|| Some(format!("toolu_{:04x}", self.tool_index)));
                        self.tool_index += 1;
                        events.push(ToolStreamEvent::ToolCallComplete(ToolCall { name, arguments, id }));
                    }
                    self.tag_buf.clear();
                    self.in_tag = false;
                    continue;
                } else {
                    if self.buf.len() > "</tool_call>".len() {
                        let mut safe = self.buf.len() - "</tool_call>".len();
                        // Ensure we don't split a multi-byte UTF-8 character
                        while safe > 0 && !self.buf.is_char_boundary(safe) {
                            safe -= 1;
                        }
                        if safe > 0 {
                            self.tag_buf.push_str(&self.buf[..safe]);
                            self.buf = self.buf[safe..].to_string();
                        }
                    }
                    break;
                }
            } else {
                if let Some(start_pos) = self.buf.find("<tool_call>") {
                    let before = &self.buf[..start_pos];
                    if !before.is_empty() { events.push(ToolStreamEvent::Text(before.to_string())); }
                    self.buf = self.buf[start_pos + "<tool_call>".len()..].to_string();
                    self.in_tag = true;
                    self.tag_buf.clear();
                    continue;
                } else {
                    let tag = "<tool_call>";
                    let mut safe_len = self.buf.len();
                    for prefix_len in (1..tag.len()).rev() {
                        if self.buf.ends_with(&tag[..prefix_len]) {
                            safe_len = self.buf.len() - prefix_len;
                            break;
                        }
                    }
                    if safe_len > 0 {
                        let safe_text = self.buf[..safe_len].to_string();
                        self.buf = self.buf[safe_len..].to_string();
                        if !safe_text.is_empty() { events.push(ToolStreamEvent::Text(safe_text)); }
                    }
                    break;
                }
            }
        }
        events
    }

    pub fn flush(&mut self) -> Vec<ToolStreamEvent> {
        let mut events = Vec::new();
        if self.in_tag {
            let mut text = String::from("<tool_call>");
            text.push_str(&self.tag_buf);
            text.push_str(&self.buf);
            if !text.is_empty() { events.push(ToolStreamEvent::Text(text)); }
        } else if !self.buf.is_empty() {
            events.push(ToolStreamEvent::Text(std::mem::take(&mut self.buf)));
        }
        self.buf.clear();
        self.tag_buf.clear();
        events
    }

    pub fn has_seen_tools(&self) -> bool {
        self.tool_index > 0
    }
}

/// Try to fix common JSON errors from small models.
fn fix_json(raw: &str) -> Option<Value> {
    // 1. Try as-is first
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        return Some(v);
    }

    let mut s = raw.to_string();

    // 2. Replace single quotes with double quotes (but not inside strings)
    //    Simple heuristic: if no double quotes found, swap all single quotes
    if !s.contains('"') {
        s = s.replace('\'', "\"");
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            return Some(v);
        }
    }

    // 3. Fix unescaped newlines inside JSON string values
    //    Strategy: find string values and escape literal newlines within them
    s = fix_unescaped_newlines(&s);
    if let Ok(v) = serde_json::from_str::<Value>(&s) {
        return Some(v);
    }

    // 4. Remove trailing commas before } or ]
    let re_trailing = regex_lite::Regex::new(r",\s*([}\]])").unwrap();
    s = re_trailing.replace_all(&s, "$1").to_string();
    if let Ok(v) = serde_json::from_str::<Value>(&s) {
        return Some(v);
    }

    // 5. Try to fix missing closing braces
    let open_braces = s.chars().filter(|&c| c == '{').count();
    let close_braces = s.chars().filter(|&c| c == '}').count();
    if open_braces > close_braces {
        for _ in 0..(open_braces - close_braces) {
            s.push('}');
        }
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            return Some(v);
        }
    }

    // 6. Try extracting name and arguments with regex as last resort
    if let Some(tc) = extract_tool_call_fuzzy(raw) {
        return Some(tc);
    }

    tracing::warn!("fix_json: could not repair: {}...", &raw[..raw.len().min(200)]);
    None
}

/// Fix unescaped literal newlines inside JSON string values.
fn fix_unescaped_newlines(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escape_next = false;
    for ch in s.chars() {
        if escape_next {
            result.push(ch);
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            result.push(ch);
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            result.push(ch);
            continue;
        }
        if in_string && ch == '\n' {
            result.push_str("\\n");
        } else if in_string && ch == '\r' {
            // skip \r, will be covered by \n
        } else if in_string && ch == '\t' {
            result.push_str("\\t");
        } else {
            result.push(ch);
        }
    }
    result
}

/// Last-resort fuzzy extraction: try to find "name" and "arguments" fields.
fn extract_tool_call_fuzzy(raw: &str) -> Option<Value> {
    let name_re = regex_lite::Regex::new(r#""name"\s*:\s*"([^"]+)""#).unwrap();
    let name = name_re.captures(raw)?.get(1)?.as_str();

    // Try to extract arguments object
    let args_re = regex_lite::Regex::new(r#""arguments"\s*:\s*(\{)"#).unwrap();
    if let Some(m) = args_re.find(raw) {
        let start = raw[m.start()..].find('{')? + m.start();
        // Find matching closing brace
        let mut depth = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut end = start;
        for (i, ch) in raw[start..].char_indices() {
            if esc { esc = false; continue; }
            if ch == '\\' && in_str { esc = true; continue; }
            if ch == '"' { in_str = !in_str; continue; }
            if !in_str {
                if ch == '{' { depth += 1; }
                if ch == '}' {
                    depth -= 1;
                    if depth == 0 { end = start + i + 1; break; }
                }
            }
        }
        if end > start {
            let args_raw = &raw[start..end];
            // Try to parse the arguments, if it fails use as-is string
            let args = serde_json::from_str::<Value>(args_raw)
                .unwrap_or_else(|_| {
                    let fixed = fix_unescaped_newlines(args_raw);
                    serde_json::from_str::<Value>(&fixed)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()))
                });
            return Some(json!({"name": name, "arguments": args}));
        }
    }

    // Fallback: no arguments found
    Some(json!({"name": name, "arguments": {}}))
}

pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find("<tool_call>") {
        let abs_start = search_from + start + "<tool_call>".len();
        if let Some(end) = text[abs_start..].find("</tool_call>") {
            let abs_end = abs_start + end;
            let inner = text[abs_start..abs_end].trim();
            if let Some(parsed) = fix_json(inner) {
                let name = parsed.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let arguments = parsed.get("arguments").map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string())).unwrap_or_else(|| "{}".to_string());
                let id = parsed.get("call_id").and_then(|i| i.as_str()).map(|s| s.to_string());
                if !name.is_empty() { calls.push(ToolCall { name, arguments, id }); }
            }
            search_from = abs_end + "</tool_call>".len();
        } else { break; }
    }
    calls
}

fn strip_tool_calls(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<tool_call>") {
        if let Some(end) = result[start..].find("</tool_call>") {
            let abs_end = start + end + "</tool_call>".len();
            result = format!("{}{}", &result[..start], &result[abs_end..]);
        } else { break; }
    }
    result
}
