import json
import sys
import os

def recover(log_file):
    log_path = os.path.join('payload_logs', log_file)
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {log_path} is not valid JSON.")
            return

    last_payload = data.get("last", [])
    if not last_payload:
        print("Error: No 'last' payload found in the log.")
        return

    messages = []
    summaries = []
    visual_memory = ""

    main_prompt_added = False

    for msg in last_payload:
        role = msg.get("role")
        content = msg.get("content", "")

        # Robustly handle cases where content is a list (e.g., Vision API or Claude Caching)
        text_parts = []
        if isinstance(content, list):
            for p in content:
                if p.get('type') == 'text':
                    text_parts.append(p.get('text', ''))
                elif p.get('type') == 'image_url':
                    text_parts.append("\n*[Recovered Image Attachment]*")
            text_content = "\n".join(text_parts).strip()
        else:
            text_content = str(content).strip()

        if role == "system":
            if "--- PERMANENT CHARACTER VISUALS & LORE ---" in text_content:
                parts = text_content.split('\n', 1)
                if len(parts) > 1:
                    visual_memory = parts[1].strip()
            elif "SYSTEM NOTE FOR MAIN MODEL" in text_content:
                continue
            elif "[REPETITION CONTROL UNIT]" in text_content:
                continue
            elif "RULES FOR THE ASSISTANT:" in text_content:
                continue
            elif "--- RELEVANT LORE (RETRIEVED CONTEXT) ---" in text_content:
                continue
            elif text_content.startswith("--- RECENT SUMMARY") or text_content.startswith("--- CONSOLIDATED ARCHIVE"):
                is_consolidated = "CONSOLIDATED ARCHIVE" in text_content
                # Strip the prefix header line safely
                parts = text_content.split('\n', 1)
                clean_content = parts[1].strip() if len(parts) > 1 else text_content

                dummy_idx = len(messages)

                # Create a visible dummy message so the user can read the recovered summary
                # right inside the chat window. The UI will mark it as "SUMMARIZED".
                messages.append({
                    "role": "assistant",
                    "content": f"*(Recovered Summary)*\n\n{clean_content}"
                })

                summaries.append({
                    "start_index": dummy_idx,
                    "end_index": dummy_idx,
                    "content": clean_content,
                    "is_consolidated": is_consolidated,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}
                })
            else:
                if not main_prompt_added:
                    # Append the main prompt or character persona prompt
                    messages.append({"role": "system", "content": text_content})
                    main_prompt_added = True
        else:
            # Append standard User and Assistant messages
            messages.append({"role": role, "content": text_content})

    output_data = {
        "messages": messages,
        "summaries": summaries,
        "visual_memory": visual_memory,
        "character_slug": None
    }

    out_name = "Recovered_" + log_file
    if not out_name.endswith('.json'):
        out_name += '.json'

    out_path = os.path.join('conversations', out_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully recovered chat to: {out_path}")
    print("Refresh your browser to see the recovered chat in the sidebar.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recover_chat.py <filename_in_payload_logs>")
    else:
        recover(sys.argv[1])