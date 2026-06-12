use serde_json::Value;
use tracing::info;

/// Rough estimate: 1 token ≈ 3 bytes for English, ≈ 2 bytes for Chinese.
/// We use a conservative 2.5 bytes/token to handle mixed content.
fn estimate_tokens(text: &str) -> usize {
    // Count chars, not bytes. Chinese chars ≈ 1 token each, English words ≈ 1 token per 4 chars.
    let chars = text.chars().count();
    let cjk = text.chars().filter(|c| *c > '\u{2E80}').count();
    // CJK chars: ~1 token each. Non-CJK: ~1 token per 4 chars.
    cjk + (chars - cjk) / 4 + 1
}

/// Truncate messages array to fit within max_tokens.
/// Keeps: system prompt (first) + last N messages that fit.
/// Returns the number of messages dropped.
pub fn truncate_messages(messages: &mut Vec<Value>, max_tokens: usize) -> usize {
    if messages.is_empty() || max_tokens == 0 {
        return 0;
    }

    // Reserve tokens for model output (at least 2048)
    let input_budget = max_tokens.saturating_sub(2048);
    if input_budget == 0 { return 0; }

    // Calculate tokens for each message
    let msg_tokens: Vec<usize> = messages.iter().map(|m| {
        let content = m.get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        estimate_tokens(content) + 4 // overhead for role, formatting
    }).collect();

    let total: usize = msg_tokens.iter().sum();
    if total <= input_budget {
        return 0; // fits, no truncation needed
    }

    // Always keep system message(s) at the front
    let mut system_end = 0;
    let mut system_tokens = 0;
    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
            system_end = i + 1;
            system_tokens += msg_tokens[i];
        } else {
            break;
        }
    }

    let remaining_budget = input_budget.saturating_sub(system_tokens);

    // Take messages from the end until budget is exceeded
    let non_system = &msg_tokens[system_end..];
    let mut keep_from = non_system.len();
    let mut used = 0;
    for (i, &tokens) in non_system.iter().enumerate().rev() {
        if used + tokens > remaining_budget {
            break;
        }
        used += tokens;
        keep_from = i;
    }

    let drop_count = keep_from;
    if drop_count == 0 {
        return 0;
    }

    // Remove old messages (keep system + recent)
    let kept_msgs: Vec<Value> = messages[..system_end].iter()
        .chain(messages[system_end + drop_count..].iter())
        .cloned()
        .collect();

    let original_len = messages.len();
    *messages = kept_msgs;
    let dropped = original_len - messages.len();

    info!("Context truncation: dropped {} old messages, ~{} tokens → ~{} tokens (budget: {})",
        dropped, total, total - (msg_tokens[system_end..system_end + drop_count].iter().sum::<usize>()), input_budget);

    dropped
}

/// Estimate if the conversation needs deep thinking.
/// Returns true if the query is complex enough to warrant reasoning.
pub fn estimate_complexity(messages: &[Value]) -> bool {
    let user_messages: Vec<&str> = messages.iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect();

    let last_user = match user_messages.last() {
        Some(msg) => *msg,
        None => return false,
    };

    let all_content: String = messages.iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");

    let mut score: i32 = 0;

    // 1. Total context size
    let total_len = all_content.len();
    if total_len > 8000 { score += 3; }
    else if total_len > 3000 { score += 2; }
    else if total_len > 1000 { score += 1; }

    // 2. Conversation depth
    let turn_count = messages.len();
    if turn_count > 6 { score += 2; }
    else if turn_count > 2 { score += 1; }

    // 3. Last user message length
    let last_len = last_user.len();
    if last_len > 2000 { score += 3; }
    else if last_len > 500 { score += 2; }
    else if last_len > 150 { score += 1; }

    // 4. Code blocks
    let code_block_count = all_content.matches("```").count() / 2;
    if code_block_count >= 3 { score += 3; }
    else if code_block_count >= 1 { score += 2; }

    // 5. Complexity keywords (CN + EN)
    let complex_keywords = [
        "analyze", "debug", "refactor", "implement", "algorithm", "optimize",
        "architecture", "design", "prove", "derive", "calculate", "explain why",
        "step by step", "trade-off", "compare", "evaluate", "fix", "error",
        "分析", "调试", "重构", "实现", "算法", "优化",
        "架构", "设计", "证明", "推导", "计算", "为什么",
        "逐步", "对比", "评估", "解释原理", "怎么做", "修复", "报错",
    ];
    let last_lower = last_user.to_lowercase();
    let keyword_hits = complex_keywords.iter()
        .filter(|kw| last_lower.contains(&kw.to_lowercase()))
        .count();
    score += keyword_hits as i32;

    // 6. Math / logic indicators
    if last_user.contains("$$") || last_user.contains("\\frac") || last_user.contains("\\sum") {
        score += 2;
    }

    // 7. System prompt length
    let system_len: usize = messages.iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.len())
        .sum();
    if system_len > 1000 { score += 2; }
    else if system_len > 300 { score += 1; }

    score >= 3
}
