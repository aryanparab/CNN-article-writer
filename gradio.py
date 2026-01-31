"""
Advanced Gradio UI for CNN Article Writer
Features: Batch processing, export, history, and more
"""

import gradio as gr
from unsloth import FastLanguageModel
import torch
import json
from datetime import datetime
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "Llama-3.2-1B-CNN-Article-Writer"

# ============================================================================
# LOAD MODEL
# ============================================================================

print("🔄 Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
print("✅ Model loaded successfully!")

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_article(rough_notes, temperature=0.7, max_length=512, top_p=0.9, 
                    system_prompt="", style="professional"):
    """Generate a single article"""
    
    if not rough_notes.strip():
        return "⚠️ Please enter some rough notes."
    
    # Style-specific system prompts
    style_prompts = {
        "professional": "You are a professional journalist. Expand rough notes into complete, well-written news articles. Maintain all facts while adding proper structure and professional language.",
        "casual": "You are a friendly journalist. Expand rough notes into engaging, conversational news articles. Keep it informative but relaxed.",
        "formal": "You are a formal news correspondent. Expand rough notes into structured, objective news articles with precise language and clear organization.",
        "investigative": "You are an investigative journalist. Expand rough notes into detailed, analytical news articles that explore context and implications."
    }
    
    if not system_prompt.strip():
        system_prompt = style_prompts.get(style, style_prompts["professional"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Expand these rough notes into a professional news article:\n\n{rough_notes}"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    article = result.split("assistant")[-1].strip() if "assistant" in result else result.strip()
    
    return article

def batch_generate(file, temperature, max_length, top_p):
    """Process multiple articles from CSV/JSON file"""
    
    if file is None:
        return "⚠️ Please upload a file", None
    
    try:
        # Read file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            if 'rough_notes' not in df.columns:
                return "⚠️ CSV must have a 'rough_notes' column", None
            notes_list = df['rough_notes'].tolist()
        
        elif file.name.endswith('.json'):
            with open(file.name, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                notes_list = [item.get('rough_notes', '') for item in data]
            else:
                return "⚠️ JSON must be a list of objects with 'rough_notes' field", None
        
        else:
            return "⚠️ Please upload a CSV or JSON file", None
        
        # Generate articles
        results = []
        for i, notes in enumerate(notes_list):
            article = generate_article(notes, temperature, max_length, top_p)
            results.append({
                'id': i + 1,
                'rough_notes': notes,
                'generated_article': article,
                'timestamp': datetime.now().isoformat()
            })
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = f"generated_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_df.to_csv(output_path, index=False)
        
        summary = f"✅ Successfully generated {len(results)} articles!\n\nSaved to: {output_path}"
        
        return summary, output_path
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def export_article(rough_notes, article, format_type):
    """Export article in different formats"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type == "TXT":
        filename = f"article_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(f"ROUGH NOTES:\n{'-'*50}\n{rough_notes}\n\n")
            f.write(f"GENERATED ARTICLE:\n{'-'*50}\n{article}\n")
    
    elif format_type == "JSON":
        filename = f"article_{timestamp}.json"
        data = {
            "rough_notes": rough_notes,
            "generated_article": article,
            "timestamp": datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format_type == "Markdown":
        filename = f"article_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write(f"# Generated Article\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Rough Notes\n\n{rough_notes}\n\n")
            f.write(f"## Article\n\n{article}\n")
    
    return filename

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="CNN Article Writer Pro", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 📰 CNN Article Writer Pro
        
        Transform rough notes into polished articles with advanced features
        """
    )
    
    with gr.Tabs():
        
        # TAB 1: Single Article Generation
        with gr.Tab("✍️ Single Article"):
            with gr.Row():
                with gr.Column():
                    notes_input = gr.Textbox(
                        label="📝 Rough Notes",
                        placeholder="• Bullet point 1\n• Bullet point 2\n• etc.",
                        lines=10
                    )
                    
                    with gr.Row():
                        style_dropdown = gr.Dropdown(
                            choices=["professional", "casual", "formal", "investigative"],
                            value="professional",
                            label="✨ Writing Style"
                        )
                        temp_slider = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="🌡️ Temperature")
                    
                    with gr.Row():
                        max_len_slider = gr.Slider(128, 1024, 512, step=64, label="📏 Max Length")
                        top_p_slider = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="🎯 Top P")
                    
                    with gr.Accordion("🔧 Custom System Prompt", open=False):
                        custom_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="Leave empty to use style preset",
                            lines=3
                        )
                    
                    generate_btn = gr.Button("🚀 Generate Article", variant="primary", size="lg")
                
                with gr.Column():
                    output_article = gr.Textbox(
                        label="📄 Generated Article",
                        lines=15,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        export_format = gr.Radio(
                            choices=["TXT", "JSON", "Markdown"],
                            value="TXT",
                            label="📥 Export Format"
                        )
                        export_btn = gr.Button("💾 Export")
                    
                    export_file = gr.File(label="Downloaded File")
            
            # Examples
            gr.Examples(
                examples=[
                    ["• Scientists testified before House Subcommittee\n• Dr. Herberman can't confirm phones are safe\n• Study found 2x cancer risk", "professional", 0.7, 512, 0.9],
                    ["• Lakers beat Warriors 112-108\n• LeBron scores 35 points\n• Game decided in final seconds", "casual", 0.7, 400, 0.9],
                ],
                inputs=[notes_input, style_dropdown, temp_slider, max_len_slider, top_p_slider],
            )
        
        # TAB 2: Batch Processing
        with gr.Tab("📦 Batch Processing"):
            gr.Markdown(
                """
                Upload a CSV or JSON file with a `rough_notes` column/field to generate multiple articles at once.
                
                **CSV format:**
                ```
                rough_notes
                "• Note 1"
                "• Note 2"
                ```
                
                **JSON format:**
                ```json
                [
                  {"rough_notes": "• Note 1"},
                  {"rough_notes": "• Note 2"}
                ]
                ```
                """
            )
            
            with gr.Row():
                with gr.Column():
                    batch_file = gr.File(label="📁 Upload CSV or JSON", file_types=[".csv", ".json"])
                    
                    with gr.Row():
                        batch_temp = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="🌡️ Temperature")
                        batch_max = gr.Slider(128, 1024, 512, step=64, label="📏 Max Length")
                    
                    batch_top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="🎯 Top P")
                    
                    batch_btn = gr.Button("🔄 Process Batch", variant="primary", size="lg")
                
                with gr.Column():
                    batch_output = gr.Textbox(label="📊 Status", lines=5)
                    batch_download = gr.File(label="📥 Download Results")
        
        # TAB 3: Help
        with gr.Tab("❓ Help"):
            gr.Markdown(
                """
                ## How to Use
                
                ### Single Article Mode
                1. Enter your rough notes (bullet points work best)
                2. Choose a writing style
                3. Adjust temperature (higher = more creative)
                4. Click "Generate Article"
                5. Export in your preferred format
                
                ### Batch Processing Mode
                1. Prepare a CSV or JSON file with your notes
                2. Upload the file
                3. Set generation parameters
                4. Click "Process Batch"
                5. Download the results CSV
                
                ### Tips for Better Results
                - ✅ Include specific names, dates, and numbers
                - ✅ Use bullet points for clarity
                - ✅ Provide context (who, what, when, where)
                - ✅ Keep facts concise but complete
                - ❌ Don't include contradictory information
                
                ### Parameters Explained
                - **Temperature**: Controls randomness (0.1 = focused, 1.0 = creative)
                - **Max Length**: Maximum tokens in output
                - **Top P**: Nucleus sampling (0.9 recommended)
                - **Style**: Preset writing styles for different tones
                
                ### Model Info
                - **Base Model**: Llama 3.2 1B
                - **Fine-tuned on**: CNN/DailyMail articles
                - **Purpose**: Educational/Research
                """
            )
    
    # Connect functions
    generate_btn.click(
        fn=generate_article,
        inputs=[notes_input, temp_slider, max_len_slider, top_p_slider, custom_prompt, style_dropdown],
        outputs=output_article
    )
    
    export_btn.click(
        fn=export_article,
        inputs=[notes_input, output_article, export_format],
        outputs=export_file
    )
    
    batch_btn.click(
        fn=batch_generate,
        inputs=[batch_file, batch_temp, batch_max, batch_top_p],
        outputs=[batch_output, batch_download]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )