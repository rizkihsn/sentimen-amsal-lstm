import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer

print("🔧 Memulai rebuild tokenizer...")

old_tokenizer_path = 'model_training/tokenizer.pkl'
new_tokenizer_path = 'model_training/tokenizer_v2.pkl'

# ===== STEP 1: Try load old tokenizer =====
print("\n📖 Loading old tokenizer...")
old_tokenizer = None
try:
    with open(old_tokenizer_path, 'rb') as f:
        old_tokenizer = pickle.load(f)
    print(f"✅ Old tokenizer loaded. Vocab size: {len(old_tokenizer.word_index)}")
except Exception as e:
    print(f"⚠️ Could not load old tokenizer: {e}")
    print("📝 Will create new tokenizer from scratch...")

# ===== STEP 2: Create or recreate tokenizer =====
print("\n🏗️ Creating/Recreating tokenizer...")

if old_tokenizer:
    # Copy dari old tokenizer
    new_tokenizer = Tokenizer(num_words=10000)
    new_tokenizer.word_index = old_tokenizer.word_index
    new_tokenizer.word_counts = old_tokenizer.word_counts
    new_tokenizer.word_docs = old_tokenizer.word_docs
    new_tokenizer.document_count = old_tokenizer.document_count
    print("✅ Tokenizer recreated from old tokenizer")
else:
    # Buat baru
    new_tokenizer = Tokenizer(num_words=10000)
    print("✅ New tokenizer created")

# ===== STEP 3: Save tokenizer baru =====
print(f"\n💾 Saving tokenizer to {new_tokenizer_path}...")
try:
    with open(new_tokenizer_path, 'wb') as f:
        pickle.dump(new_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Tokenizer saved: {new_tokenizer_path}")
except Exception as e:
    print(f"❌ Error saving tokenizer: {e}")
    exit(1)

# ===== STEP 4: Test load =====
print("\n🧪 Testing tokenizer...")
try:
    with open(new_tokenizer_path, 'rb') as f:
        test_tokenizer = pickle.load(f)
    
    # Test texts_to_sequences
    test_text = "ini adalah sentimen positif"
    test_seq = test_tokenizer.texts_to_sequences([test_text])
    print(f"✅ Tokenizer works! Test sequence: {test_seq}")
except Exception as e:
    print(f"❌ Error testing tokenizer: {e}")
    exit(1)

print("\n" + "="*50)
print("✅ TOKENIZER REBUILD SELESAI!")
print("="*50)
print(f"\n📝 LANGKAH SELANJUTNYA:")
print(f"1. Update app.py - ganti:")
print(f"   'model_training/tokenizer.pkl'")
print(f"   menjadi:")
print(f"   'model_training/tokenizer_v2.pkl'")
print(f"\n2. Jalankan: streamlit run app.py")
print("="*50)