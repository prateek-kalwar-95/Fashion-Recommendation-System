# Vibe Matcher: AI-Powered Fashion Recommendation System
# Author: Prateek Kalwar

"""
Introduction: Why AI at Nexora?
I'm passionate about leveraging AI to create personalized user experiences that feel
intuitive and human. Nexora's focus on building intelligent systems that understand
user intent resonates with my belief that the best technology disappears into
seamless experiences. This vibe matcher demonstrates my approach: using embeddings
to capture semantic meaning rather than keyword matching, creating recommendations
that truly understand what users are looking for.
"""


#RUN the below command and then run the code and also add the gemini API_KEY
# install these libraries: pip install google-generativeai numpy pandas matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# For Google Gemini API
# !pip install google-generativeai
import google.generativeai as genai

print("All libraries imported successfully!")


def create_fashion_dataset() -> pd.DataFrame:
    """
    Create a mock dataset of fashion products with descriptions and vibe tags.
    """
    products = [
        {
            "name": "Boho Maxi Dress",
            "description": "Flowy maxi dress with earthy tones, perfect for festival vibes and outdoor concerts. Features floral patterns and relaxed fit.",
            "vibes": ["boho", "casual", "earthy", "festival"]
        },
        {
            "name": "Urban Leather Jacket",
            "description": "Sleek black leather jacket with modern cut. Edgy and sophisticated for city nightlife and urban adventures.",
            "vibes": ["urban", "edgy", "modern", "sophisticated"]
        },
        {
            "name": "Cozy Oversized Sweater",
            "description": "Chunky knit oversized sweater in neutral beige. Warm, comfortable, and perfect for lazy Sunday mornings with coffee.",
            "vibes": ["cozy", "comfortable", "casual", "relaxed"]
        },
        {
            "name": "Energetic Neon Tracksuit",
            "description": "Bold neon green athletic tracksuit. High-energy design for fitness enthusiasts and those who love to stand out.",
            "vibes": ["energetic", "athletic", "bold", "sporty"]
        },
        {
            "name": "Elegant Silk Blouse",
            "description": "Luxurious cream silk blouse with delicate details. Sophisticated and timeless for business meetings or dinner dates.",
            "vibes": ["elegant", "sophisticated", "timeless", "professional"]
        },
        {
            "name": "Vintage Denim Jacket",
            "description": "Classic light-wash denim jacket with retro patches. Nostalgic and versatile for everyday casual wear.",
            "vibes": ["vintage", "casual", "retro", "nostalgic"]
        },
        {
            "name": "Minimalist White Tee",
            "description": "Simple, clean white cotton t-shirt. Modern minimalist design that pairs with everything for effortless style.",
            "vibes": ["minimalist", "clean", "modern", "versatile"]
        },
        {
            "name": "Romantic Lace Dress",
            "description": "Delicate pink lace dress with feminine details. Soft, romantic, and perfect for garden parties or date nights.",
            "vibes": ["romantic", "feminine", "delicate", "elegant"]
        },
        {
            "name": "Street Style Cargo Pants",
            "description": "Oversized olive green cargo pants with multiple pockets. Urban streetwear aesthetic for the fashion-forward.",
            "vibes": ["urban", "streetwear", "casual", "trendy"]
        },
        {
            "name": "Bohemian Fringe Vest",
            "description": "Suede fringe vest in tan brown. Free-spirited bohemian style perfect for music festivals and artistic events.",
            "vibes": ["boho", "artistic", "festival", "free-spirited"]
        }
    ]

    df = pd.DataFrame(products)
    print(f"Created dataset with {len(df)} fashion products")
    return df


# Create the dataset
fashion_df = create_fashion_dataset()

# Display the dataset
print("\nFashion Product Dataset:")
print("=" * 80)
for idx, row in fashion_df.iterrows():
    print(f"\n{idx + 1}. {row['name']}")
    print(f"   Description: {row['description']}")
    print(f"   Vibes: {', '.join(row['vibes'])}")

#add you openAi or gemini API_KEY
genai.configure(api_key="API_KEY")

def get_embedding_gemini(text: str, model: str = "models/embedding-001") -> List[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"  # or "retrieval_query" for queries
    )
    return result['embedding']


def generate_mock_embedding(text: str, dimension: int = 768) -> np.ndarray:
    np.random.seed(hash(text) % (2 ** 32))
    embedding = np.random.randn(dimension)

    # Add semantic patterns based on keywords (simple simulation)
    keywords = {
        'urban': 0.3, 'city': 0.3, 'modern': 0.25, 'edgy': 0.35,
        'cozy': 0.4, 'comfortable': 0.4, 'warm': 0.35, 'soft': 0.3,
        'boho': 0.45, 'festival': 0.4, 'artistic': 0.35, 'free': 0.3,
        'elegant': 0.5, 'sophisticated': 0.45, 'luxurious': 0.4,
        'energetic': 0.55, 'bold': 0.4, 'vibrant': 0.45, 'athletic': 0.4,
        'romantic': 0.35, 'delicate': 0.3, 'feminine': 0.35,
        'casual': 0.25, 'relaxed': 0.3, 'simple': 0.2
    }

    text_lower = text.lower()
    for keyword, weight in keywords.items():
        if keyword in text_lower:
            # Add a pattern to specific dimensions
            start_idx = hash(keyword) % (dimension - 100)
            embedding[start_idx:start_idx + 100] += weight

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def generate_embeddings_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerating embeddings for products...")
    embeddings = []

    for idx, row in df.iterrows():
        embedding = generate_mock_embedding(row['description'])
        embeddings.append(embedding)
        print(f"  ✓ Generated embedding for: {row['name']}")

    df['embedding'] = embeddings
    print(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

    return df


# Generate embeddings
fashion_df = generate_embeddings_for_dataset(fashion_df)


#vector similarity search
def compute_cosine_similarity(query_embedding: np.ndarray,
                              product_embeddings: List[np.ndarray]) -> np.ndarray:

    product_matrix = np.vstack(product_embeddings)

    query_matrix = query_embedding.reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query_matrix, product_matrix)[0]

    return similarities


def find_top_matches(query: str,
                     df: pd.DataFrame,
                     top_k: int = 3,
                     threshold: float = 0.5) -> List[Dict]:

    query_embedding = generate_mock_embedding(query)

    similarities = compute_cosine_similarity(query_embedding, df['embedding'].tolist())
    df_temp = df.copy()
    df_temp['similarity'] = similarities

    # Sort by similarity
    df_temp = df_temp.sort_values('similarity', ascending=False)

    # Get top matches
    top_matches = []
    for idx, row in df_temp.head(top_k).iterrows():
        match = {
            'rank': len(top_matches) + 1,
            'name': row['name'],
            'description': row['description'],
            'vibes': row['vibes'],
            'similarity_score': row['similarity'],
            'match_quality': 'Excellent' if row['similarity'] > 0.8 else
            'Good' if row['similarity'] > 0.7 else
            'Fair' if row['similarity'] > threshold else
            'Weak'
        }
        top_matches.append(match)

    # Handle edge case: no good matches
    if all(m['similarity_score'] < threshold for m in top_matches):
        print(f"⚠️  No strong matches found (all below threshold {threshold})")
        print("💡 Suggestion: Try broader or different query terms")

    return top_matches


def display_matches(query: str, matches: List[Dict]):
    print(f"\n{'=' * 80}")
    print(f"🔍 SEARCH QUERY: '{query}'")
    print(f"{'=' * 80}\n")

    for match in matches:
        print(f"#{match['rank']} {match['name']} "
              f"[Score: {match['similarity_score']:.3f} - {match['match_quality']}]")
        print(f"   {match['description']}")
        print(f"   Vibes: {', '.join(match['vibes'])}")
        print()


def run_test_queries(df: pd.DataFrame, queries: List[str]) -> Dict:
    results = {
        'queries': [],
        'latencies': [],
        'avg_scores': [],
        'match_qualities': []
    }

    for query in queries:
        print(f"\n{'#' * 80}")
        print(f"Testing Query: '{query}'")
        print(f"{'#' * 80}")

        # Measure latency
        start_time = time.time()
        matches = find_top_matches(query, df, top_k=3)
        latency = time.time() - start_time

        # Display results
        display_matches(query, matches)

        # Collect metrics
        results['queries'].append(query)
        results['latencies'].append(latency)
        results['avg_scores'].append(np.mean([m['similarity_score'] for m in matches]))
        results['match_qualities'].append([m['match_quality'] for m in matches])

        print(f"⏱️  Query latency: {latency * 1000:.2f}ms")

    return results


# Define test queries
test_queries = [
    "energetic urban chic"
    "energetic bold vibrant athletic tracksuit for fitness enthusiasts",
    "elegant sophisticated luxurious blouse for professional business"
    "boho festival artistic free-spirited bohemian style with relaxed vibes"
]
# Run tests
print("\n" + "=" * 80)
print("STARTING EVALUATION WITH 4 TEST QUERIES")
print("=" * 80)

test_results = run_test_queries(fashion_df, test_queries)


def plot_metrics(results: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Latency
    ax1 = axes[0]
    bars = ax1.bar(range(len(results['queries'])),
                   [l * 1000 for l in results['latencies']],
                   color='steelblue', alpha=0.7)
    ax1.set_xlabel('Query Number', fontsize=11)
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_title('Query Latency Performance', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(results['queries'])))
    ax1.set_xticklabels([f"Q{i + 1}" for i in range(len(results['queries']))])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}ms',
                 ha='center', va='bottom', fontsize=9)

    # Plot 2: Average Similarity Scores
    ax2 = axes[1]
    bars = ax2.bar(range(len(results['queries'])),
                   results['avg_scores'],
                   color='coral', alpha=0.7)
    ax2.set_xlabel('Query Number', fontsize=11)
    ax2.set_ylabel('Average Similarity Score', fontsize=11)
    ax2.set_title('Average Match Quality', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(results['queries'])))
    ax2.set_xticklabels([f"Q{i + 1}" for i in range(len(results['queries']))])
    ax2.axhline(y=0.7, color='green', linestyle='--', label='Good Threshold (0.7)', alpha=0.6)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Fair Threshold (0.5)', alpha=0.6)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('vibe_matcher_metrics.png', dpi=300, bbox_inches='tight')
    print("\n✅ Metrics visualization saved as 'vibe_matcher_metrics.png'")
    plt.show()


# Generate visualizations
plot_metrics(test_results)

# Print summary statistics
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Total queries tested: {len(test_results['queries'])}")
print(f"Average latency: {np.mean(test_results['latencies']) * 1000:.2f}ms")
print(f"Min latency: {np.min(test_results['latencies']) * 1000:.2f}ms")
print(f"Max latency: {np.max(test_results['latencies']) * 1000:.2f}ms")
print(f"Average similarity score: {np.mean(test_results['avg_scores']):.3f}")
print(
    f"\nQueries with 'Good' matches (>0.7): {sum(1 for s in test_results['avg_scores'] if s > 0.7)}/{len(test_results['queries'])}")


