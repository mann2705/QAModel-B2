import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class LanguageModelSetup:
    def __init__(self, vocab_size_range=(100, 5000)):
        self.vocab_size_range = vocab_size_range
#This function is used only for testing
    def generate_synthetic_llm_data(self, num_samples=100):
        general_statements = [
            "The sky is blue and often clear during the day.",
            "Programming languages allow us to communicate with computers.",
            "Water boils at 100 degrees Celsius under standard conditions.",
            "Artificial Intelligence is transforming multiple industries.",
            "Mathematics is essential for understanding the world around us."
        ]
        
        questions = [
            "What is the capital of France?", "How does gravity work?",
            "Who invented the light bulb?", "Why do leaves change color in autumn?",
            "What are the primary colors?"
        ]
        
        dialogues = [
            "Person A: Hello, how are you? Person B: I'm fine, thank you!",
            "Person A: Do you know any good restaurants nearby? Person B: Yes, there's a great one just around the corner.",
            "Person A: What time is the meeting tomorrow? Person B: It starts at 10 AM sharp.",
            "Person A: Could you help me with my project? Person B: Sure, I'd be happy to help."
        ]
        
        facts = [
            "The speed of light is approximately 299,792 kilometers per second.",
            "Honey never spoils and can be edible for centuries.",
            "Mount Everest is the highest mountain in the world.",
            "Octopuses have three hearts and blue blood.",
            "Bananas are berries, but strawberries are not."
        ]
        
        descriptions = [
            "The forest was dense with trees, their leaves creating a thick canopy that filtered the sunlight.",
            "The bustling city was filled with the sounds of cars, chatter, and the occasional bird song.",
            "The ocean waves crashed against the shore, a soothing rhythm that relaxed the visitors.",
            "The library was quiet and peaceful, with rows of books stretching as far as the eye could see.",
            "The mountain trail was steep and winding, challenging hikers with its rugged terrain."
        ]
        
        text_data = []
        
        for _ in range(num_samples):
            category = random.choice([general_statements, questions, dialogues, facts, descriptions])
            text_data.append(random.choice(category))
        
        return text_data

    def tokenize_and_create_vocab(self, text_data):
        # Create vocabulary from tokenized data
        words = set()
        for text in text_data:
            words.update(text.split())
        
        vocab = list(words)
        
        # Check vocabulary size and warn if outside specified range
        if not (self.vocab_size_range[0] <= len(vocab) <= self.vocab_size_range[1]):
            print(f"Warning: Vocabulary size ({len(vocab)}) is out of the specified range {self.vocab_size_range}.")
        
        return vocab

    def build_model_architecture(self):
        # Simple Transformer model structure
        model_architecture = {
            "Embedding Layer": {"output_dim": 256},
            "Multi-Head Self-Attention": {"num_heads": 8},
            "Feedforward Layers": {"hidden_dim": 512},
            "Positional Encoding": {"embedding_dim": 256}
        }
        return model_architecture

    def naive_bayes_classifier(self, text_data, use_tfidf=True):
        # Generate labels for simplicity (1 for half of data, 0 for other half)
        labels = [1 if i < len(text_data) // 2 else 0 for i in range(len(text_data))]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.3, random_state=42)
        
        if use_tfidf:
            vectorizer = TfidfVectorizer(max_features=self.vocab_size_range[1])
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)
        else:
            X_train = [" ".join(text.split()[:self.vocab_size_range[1]]) for text in X_train]
            X_test = [" ".join(text.split()[:self.vocab_size_range[1]]) for text in X_test]
        
        # Train Naive Bayes classifier
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        
        # Predict and calculate accuracy
        y_pred = nb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return nb_model, accuracy

# Initialize the LanguageModelSetup
lm_setup = LanguageModelSetup(vocab_size_range=(10, 100))

# Generate a synthetic dataset
text_data = lm_setup.generate_synthetic_llm_data(num_samples=100)
print("Sample of Generated Data:", text_data[:5])

# Tokenization and vocabulary creation
vocab = lm_setup.tokenize_and_create_vocab(text_data)
print("Vocabulary Size:", len(vocab))
print("Sample of Vocabulary:", vocab[:10])

# Build Transformer Model Architecture
transformer_model = lm_setup.build_model_architecture()
print("Transformer Model Architecture:", transformer_model)

# Naive Bayes Classification
nb_model, accuracy = lm_setup.naive_bayes_classifier(text_data, use_tfidf=True)
print("Naive Bayes Model Accuracy:", accuracy)