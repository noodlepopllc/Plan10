import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def main():
    import argparse, os
    os.environ['BATCH'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prompt', type=str, default="Write a long descriptive caption for this image in a formal tone.", help='prompt')
    parser.add_argument('-I', '--input', type=str, default='')
    args = parser.parse_args()
    IMAGE_PATH = args.input
    PROMPT = args.prompt
    MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    # -----------------------------------------------------------------------------
    # 1. CONFIGURE BITSANDBYTES (4-bit Quantization)
    # -----------------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        #load_in_4bit=True,
        #bnb_4bit_quant_type="nf4",              # Normal Float 4-bit (best for LLMs)
        #bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for the quantized weights
        #bnb_4bit_use_double_quant=True,         # Further reduces memory footprint
        load_in_8bit=True,
    )

    # Load processor normally (no changes here)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # 2. LOAD MODEL WITH QUANTIZATION
    # Note: We remove torch_dtype="bfloat16" and device_map=0, replacing them with 
    # the quantization config and "auto" device mapping.
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    llava_model.eval()

    with torch.no_grad():
        # Load image
        image = Image.open(IMAGE_PATH)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": PROMPT,
            },
        ]

        # Format the conversation
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
        
        # CRITICAL: The vision encoder still expects bfloat16, so we cast pixel_values
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
        print(caption)

if __name__ == '__main__':
    main()