import json
import random
import os
from typing import List, Dict, Any

# Target chat template: Qwen2.5 ChatML format
# <|im_start|>system\n{system_prompt}<|im_end|>\n
# <|im_start|>user\n{instruction}<|im_end|>\n
# <|im_start|>assistant\n{response}<|im_end|>

SYSTEM_PROMPTS = {
    "grounded": "You are an offline assistance agent. Answer the user's question using ONLY the provided context. If the answer cannot be found in the context, state: 'I am an offline assistant and that information is not in my local guide.'",
    "tutor": "You are a language tutor. Help the user learn local vocabulary, translate phrases, and ask one simple interactive follow-up question.",
    "emergency": "You are an emergency response assistant. Provide direct, non-hallucinated instructions. State what to do, who to contact, and prioritize immediate safety."
}

# --- TEMPLATE SEED DATA ---
FACILITIES = ["shelter", "community clinic", "food bank", "legal aid office", "language school", "transit hub", "refugee support center", "daycare center"]
STREETS = ["Elm Street", "Oak Avenue", "Pine Boulevard", "Maple Crossing", "Cedar Way", "Birch Lane", "Willow Court", "Spruce Road"]
HOURS = ["8:00 AM to 6:00 PM daily", "9:00 AM to 5:00 PM on weekdays", "open 24/7", "10:00 AM to 4:00 PM, closed Sundays"]
SERVICES = ["meals and emergency bedding", "basic health screenings", "non-perishable groceries", "asylum application assistance", "survival language classes", "free transit passes", "family reunification registration", "supervised childcare"]

LANGUAGES = ["Spanish", "French", "Arabic", "Pashto", "Dari", "Ukrainian", "Somali", "Mandarin"]
PHRASES = [
    ("Where is the hospital?", "¿Dónde está el hospital?", "Dehn-deh ehs-tah el ohs-pee-tahl"),
    ("I need help.", "J'ai besoin d'aide.", "Zhay buh-zwan ded"),
    ("Water, please.", "Ma'an, fadlak.", "Mah-an fad-lahk"),
    ("Where can I sleep?", "Zama khob khana cherta da?", "Zah-mah khohb khah-nah cher-tah dah"),
    ("Where is the market?", "Kuja bazar ast?", "Koo-jah bah-zahr ast"),
    ("Is it safe here?", "Tse bezpechno tut?", "Tseh bez-pehch-noh toot"),
    ("I am lost.", "Waan lumay.", "Wahn loo-my"),
    ("Thank you.", "Xie xie.", "Shyeh shyeh")
]

EMERGENCIES = [
    ("They took my passport", "Go immediately to your nearest embassy or consular office. Do not pay any unofficial agents. Contact local legal aid for emergency documentation processing."),
    ("I don't have my medicine", "Go directly to the nearest community clinic or hospital emergency room. Present any empty containers or prescriptions. Ask for the emergency medical triage officer."),
    ("I am being followed", "Move immediately to a well-lit, public location such as a transit hub, market, or police station. Do not head toward an isolated shelter. Alert any staff or security personnel present."),
    ("I have no shelter tonight", "Go directly to the Emergency Shelter on Cedar Way. They accept direct walk-ins and are open 24/7. Food and basic medical staff are available on-site.")
]


def generate_grounded_dataset(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        facility = FACILITIES[i % len(FACILITIES)]
        street = STREETS[(i * 3) % len(STREETS)]
        hour = HOURS[(i * 7) % len(HOURS)]
        service = SERVICES[(i * 11) % len(SERVICES)]
        
        context = f"The local {facility} is located at {i+100} {street}. Operating hours are {hour}. Key services offered include {service}."
        
        # Positive grounded questions
        questions = [
            f"Where is the {facility} and what services do they provide?",
            f"Can you give me the address and operating hours for the {facility}?",
            f"I need {service}. Where should I go and when is it open?"
        ]
        
        chosen_q = questions[i % len(questions)]
        answer = f"The local {facility} is located at {i+100} {street}. It is open {hour}, and provides {service}."
        
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["grounded"]},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {chosen_q}"},
                {"role": "assistant", "content": answer}
            ]
        })
    return samples


def generate_out_of_bounds_dataset(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        facility = FACILITIES[i % len(FACILITIES)]
        street = STREETS[(i * 2) % len(STREETS)]
        hour = HOURS[(i * 5) % len(HOURS)]
        service = SERVICES[(i * 9) % len(SERVICES)]
        
        context = f"The local {facility} is located at {i+100} {street}. Operating hours are {hour}. Key services offered include {service}."
        
        # Out-of-bounds questions (asking details completely absent from context)
        adversarial_questions = [
            f"What is the personal phone number or email of the manager at {facility}?",
            f"What is the funding source and annual budget of the center on {street}?",
            f"How many legal disputes did the {facility} settle last year?",
            f"Can I get a coupon code for transit near {street}?"
        ]
        
        chosen_q = adversarial_questions[i % len(adversarial_questions)]
        
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["grounded"]},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {chosen_q}"},
                {"role": "assistant", "content": "I am an offline assistant and that information is not in my local guide."}
            ]
        })
    return samples


def generate_tutor_dataset(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        lang = LANGUAGES[i % len(LANGUAGES)]
        eng, native, pron = PHRASES[i % len(PHRASES)]
        
        instruction = f"How do I say '{eng}' in {lang} and pronounce it?"
        
        response = (
            f"In {lang}, you say: '{native}'.\n"
            f"Pronunciation guide: {pron}.\n\n"
            f"Would you like to try saying this aloud, or should we practice another common survival phrase?"
        )
        
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["tutor"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        })
    return samples


def generate_emergency_dataset(count: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(count):
        scenario, advice = EMERGENCIES[i % len(EMERGENCIES)]
        
        instruction = f"Help me, {scenario.lower()}! What do I do?"
        response = f"I understand you are in a difficult situation. Here is what you should do immediately:\n1. {advice}\n2. Prioritize your safety and do not engage with unknown individuals."
        
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["emergency"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        })
    return samples


def main():
    print("[Datapack] Synthesizing domain datasets...")
    
    # 2000 total target samples
    grounded_data = generate_grounded_dataset(600)
    oob_data = generate_out_of_bounds_dataset(600)
    tutor_data = generate_tutor_dataset(400)
    emergency_data = generate_emergency_dataset(400)
    
    all_samples = grounded_data + oob_data + tutor_data + emergency_data
    
    # Deterministic shuffle to blend tasks across batches
    random.seed(42)
    random.shuffle(all_samples)
    
    # 90/10 Train/Val Split
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"[Datapack] Split complete: {len(train_samples)} Train, {len(val_samples)} Validation")
    
    os.makedirs("./datapack", exist_ok=True)
    
    # Write to JSONL
    with open("./datapack/train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open("./datapack/val.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print("[Datapack] Files written to ./datapack/train.jsonl and ./datapack/val.jsonl")


if __name__ == "__main__":
    main()