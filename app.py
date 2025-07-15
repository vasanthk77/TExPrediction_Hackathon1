import streamlit as st
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import random
import os
import pandas as pd
import io

@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-7",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipe()

STYLE_TEMPLATES = {
    "3D Product": {
        "prompt": "cinematic 3D render of {product}, studio lighting, ultra-detailed, 8K",
        "colors": ["#2a52be", "#ffffff"]
    },
    "UGC Social": {
        "prompt": "smartphone photo of {product}, authentic, bright colors, social media post",
        "colors": ["#ff4757", "#f1f2f6"]
    },
    "Luxury": {
        "prompt": "luxury {product}, gold accents, dark moody lighting, marble background",
        "colors": ["#000000", "#d4af37"]
    }
}

ASPECT_RATIOS = {
    "Instagram": (512, 512),
    "Twitter": (600, 338),
    "Facebook": (600, 450),
    "LinkedIn": (600, 313),
    "TikTok": (384, 680),
}

def enhance_prompt(brand, product, audience, style_key):
    base = STYLE_TEMPLATES[style_key]["prompt"].format(product=product)
    return (f"{base}, high-quality marketing photo for {brand}, appealing to {audience.lower()}, "
            "award-winning composition, emotionally engaging, branding focus")

def generate_platform_hashtags(brand, audience, platform):
    brand_tag = f"#{brand.replace(' ', '')}"
    audience_tag = f"#{audience.replace(' ', '')}"
    tags_by_platform = {
        "Instagram": "#InstaDaily #StyleGoals #OOTD",
        "Twitter": "#TrendingNow #Ad #Promo",
        "Facebook": "#Community #Deals #FamilyFaves",
        "LinkedIn": "#BrandStory #Productivity #BusinessStyle",
        "TikTok": "#ForYou #FYP #TikTokMadeMeBuyIt",
    }
    extra_tags = tags_by_platform.get(platform, "#AdGenAI")
    return f"{brand_tag} {audience_tag} {extra_tags}"

def generate_copy(product, platform):
    captions_by_platform = {
        "Instagram": (f"\U0001F525 Style meets comfort with {product}!", f"Turn heads with {product} \U0001F4AF\u2728"),
        "Twitter": (f"{product} just dropped. \U0001F95F", f"Fast, fresh, flawless. #Ad"),
        "Facebook": (f"Discover the all-new {product}", f"{product} made for you — crafted for every step."),
        "LinkedIn": (f"{product} – Engineered for professionals", f"Boost your productivity and confidence with every step."),
        "TikTok": (f"POV: You rock {product} \U0001F60E", f"When your shoes do the talking \U0001F483\U0001F57A"),
    }
    headline, caption = captions_by_platform.get(platform, (f"Unleash Your Style with {product}", f"Step into confidence with {product}"))
    cta = random.choice(["Shop Now", "Learn More", "Try Today", "Limited Offer"])
    return headline, caption, cta

def generate_ads(brand, product, audience, style, ad_count, platform):
    images = []
    texts = []
    prompt = enhance_prompt(brand, product, audience, style)
    variant_labels = ['A', 'B', 'C', 'D', 'E']

    for i in range(ad_count):
        generator = torch.Generator("cuda").manual_seed(np.random.randint(100))
        image = pipe(prompt, generator=generator, num_inference_steps=12).images[0]
        if platform in ASPECT_RATIOS:
            image = image.resize(ASPECT_RATIOS[platform])

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        template = STYLE_TEMPLATES[style]
        draw.text((20, 20), brand, fill=template["colors"][0], font=font)
        draw.text((20, 70), product, fill=template["colors"][1], font=font)

        if os.path.exists("logo.png"):
            try:
                logo = Image.open("logo.png").convert("RGBA").resize((80, 80))
                image.paste(logo, (image.width - 100, 20), logo)
            except Exception as e:
                print("Logo error:", e)

        ctr_val = round(np.random.uniform(4.0, 7.0), 1)
        engagement_val = np.random.randint(8, 15)
        cpc_val = round(np.random.uniform(0.15, 0.30), 2)

        headline, caption, cta = generate_copy(product, platform)
        hashtags = generate_platform_hashtags(brand, audience, platform)
        variant = variant_labels[i] if i < len(variant_labels) else chr(65 + i)

        images.append(image)
        texts.append({
            "Variant": variant,
            "CTR": ctr_val,
            "Engagement": engagement_val,
            "CPC": cpc_val,
            "Headline": headline,
            "Caption": caption,
            "CTA": cta,
            "Hashtags": hashtags
        })

    return images, texts, prompt

# === Streamlit UI ===
st.set_page_config(layout="wide")
mode = st.sidebar.radio("\U0001F317 Choose Theme", ["Light", "Dark"], index=1)
if mode == "Dark":
    st.markdown("""<style>body { background-color: #111111; color: white; }</style>""", unsafe_allow_html=True)

st.title("\U0001F3AF AI AdGenie — Ad Variant Generator")

with st.sidebar:
    brand = st.text_input("Brand Name", "Nike")
    product = st.text_input("Product/Service", "Air Max Shoes")
    audience = st.selectbox("Target Audience", ["Gen Z", "Millennials", "Parents"])
    style = st.radio("Ad Style", list(STYLE_TEMPLATES.keys()), index=1)
    platform = st.selectbox("Platform", list(ASPECT_RATIOS.keys()))
    ad_count = st.slider("Number of Ad Variants", 1, 5, 3)
    go = st.button("\u2728 Generate Ads")

if go:
    with st.spinner("Generating ads..."):
        images, metrics, full_prompt = generate_ads(brand, product, audience, style, ad_count, platform)

    st.subheader("\U0001F4F8 Ad Variants")
    for i in range(0, len(images), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(images):
                with cols[j]:
                    img_bytes = io.BytesIO()
                    images[i + j].save(img_bytes, format="PNG")
                    byte_data = img_bytes.getvalue()

                    st.download_button(
                        label="⬇️ Download",
                        data=byte_data,
                        file_name=f"ad_variant_{metrics[i + j]['Variant']}.png",
                        mime="image/png",
                        key=f"dl_{i+j}"
                    )

                    st.image(images[i + j], caption=f"Variant {metrics[i + j]['Variant']}", width=300)
                    st.markdown(f"**Headline**: {metrics[i + j]['Headline']}")
                    st.markdown(f"**Caption**: {metrics[i + j]['Caption']}")
                    st.markdown(f"**CTA**: {metrics[i + j]['CTA']}")
                    st.markdown(f"**CTR**: {metrics[i + j]['CTR']}% | **Engagement**: {metrics[i + j]['Engagement']}% | **CPC**: ${metrics[i + j]['CPC']}")
                    st.markdown(f"**Hashtags**: {metrics[i + j]['Hashtags']}")

    st.subheader("\U0001F50D Final Prompt Used")
    st.code(full_prompt, language="text")

    st.subheader("\U0001F4CA Compare Ad Variants")
    df = pd.DataFrame(metrics)
    st.table(df[["Variant", "CTR", "Engagement", "CPC"]])

    best_variant = max(metrics, key=lambda x: x["CTR"])
    st.success(f"\U0001F3C6 **Best Performing Variant:** {best_variant['Variant']} with CTR: {best_variant['CTR']}%")
