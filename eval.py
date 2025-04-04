from medical_chatbot import MedicalChatbot
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Initialize chatbot
chatbot = MedicalChatbot()

# Basic-level mental health questions + expected answers
test_data = [
    {
        "question": "I feel anxious all the time. What should I do?",
        "reference": "Try relaxation techniques like deep breathing or mindfulness. If anxiety continues, speak to a mental health professional."
    },
    {
        "question": "I have trouble sleeping at night. Any advice?",
        "reference": "Maintain a consistent sleep schedule, avoid screens before bed, and create a relaxing bedtime routine."
    },
    {
        "question": "I feel sad even when things are going well. Is that normal?",
        "reference": "It’s okay to feel that way sometimes. If the sadness persists, consider talking to a therapist for support."
    },
    {
        "question": "I don't feel motivated to do anything. What can I do?",
        "reference": "Start small and set simple goals. Sometimes a routine and talking to someone you trust can help improve motivation."
    },
    {
        "question": "How can I manage stress better?",
        "reference": "Exercise, proper sleep, journaling, and talking to someone about your feelings can help manage stress."
    },
    {
        "question": "I feel overwhelmed with studies. Any tips?",
        "reference": "Break tasks into smaller steps, take breaks, and reach out to someone if it becomes too much."
    },
    {
        "question": "Is it okay to cry when I'm stressed?",
        "reference": "Yes, crying is a natural emotional release. It's okay to feel overwhelmed sometimes."
    },
    {
        "question": "How do I help a friend who seems depressed?",
        "reference": "Listen without judgment, encourage them to seek professional help, and remind them they're not alone."
    },
    {
        "question": "Why do I feel lonely even with people around?",
        "reference": "Loneliness can stem from emotional disconnection. It may help to talk to someone you trust or a therapist."
    },
    {
        "question": "Is it normal to have mood swings?",
        "reference": "Occasional mood swings are normal, but if they interfere with daily life, consider speaking with a professional."
    }
]

# Initialize evaluation scorers
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Containers for scores
bleu_scores = []
rouge1_scores = []
rougeL_scores = []
bertscore_preds = []
bertscore_refs = []

# Evaluation loop
for sample in test_data:
    question = sample["question"]
    reference = sample["reference"]
    
    generated = chatbot.generate_response(question)

    print(f"\n[Question]   {question}")
    print(f"[Generated]  {generated}")
    print(f"[Reference]  {reference}")

    # BLEU
    bleu = sentence_bleu([reference.split()], generated.split())
    bleu_scores.append(bleu)

    # ROUGE
    rouge_scores = rouge.score(reference, generated)
    rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
    rougeL_scores.append(rouge_scores["rougeL"].fmeasure)

    # For BERTScore
    bertscore_preds.append(generated)
    bertscore_refs.append(reference)

# BERTScore
P, R, F1 = bert_score(bertscore_preds, bertscore_refs, lang="en", model_type="bert-base-uncased")

# Display average scores
print("\n================ Evaluation Results ================")
print(f"ROUGE-1 F1 (avg):     {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-L F1 (avg):     {sum(rougeL_scores)/len(rougeL_scores):.4f}")
print(f"BERTScore F1 (avg):   {F1.mean().item():.4f}")
print("====================================================")
