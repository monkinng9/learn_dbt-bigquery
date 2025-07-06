from dotenv import load_dotenv
import os

# Load environment variables from the specified path
load_dotenv(dotenv_path='/home/sparky/.dbt/.env')

import polars as pl
from openai import OpenAI
import pickle
import numpy as np
import json


# ## Read Raw Data

raw_path = '/opt/spark/work-dir/data/exists_data/raw_embedding_demo.csv'

raw_df = pl.read_csv(source=raw_path, has_header=True)
raw_df.head()

em_01 = raw_df.filter(pl.col("ecommerce_name") == 'ecommerce_01')
em_02 = raw_df.filter(pl.col("ecommerce_name") == 'ecommerce_02')

# ## Setup & Helper Function

# Ensure OPENAI_API_KEY environment variable is set
client = OpenAI()

# Define function: generate text embeddings
def generate_embeddings(text):
    """
    Generate embeddings for input text using OpenAI's text-embedding-3-small model.

    Args:
        text (str or list[str]): The text or list of texts to embed.

    Returns:
        list or list[list]: A list of embeddings, or a list of lists of embeddings if input was a list.
    """
    if not text:
        return []
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embeddings = [record.embedding for record in response.data]
    return embeddings if isinstance(text, list) else embeddings[0]

# Set embedding cache path
embedding_cache_path = "embeddings_cache.pkl"

# Try to load cached embeddings
try:
    with open(embedding_cache_path, 'rb') as f:
        embedding_cache = pickle.load(f)
except FileNotFoundError:
    embedding_cache = {}

# Define function: obtain text embeddings through caching mechanism
def embedding_from_string(string: str, embedding_cache=embedding_cache) -> list:
    """
    Get embedding for given text, using cache mechanism to avoid recomputation.

    Args:
        string (str): The input text to get the embedding for.
        embedding_cache (dict): A dictionary used as a cache for embeddings.

    Returns:
        list: The embedding vector for the input string.
    """
    if string not in embedding_cache.keys():
        embedding_cache[string] = generate_embeddings(string)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[string]


# ## Embed base product name
# Define a dynamic batch size. You can adjust this value based on API limits and performance.
BATCH_SIZE = 60

print("Preparing base by generating embeddings for all base products...")
# Get all product names from the base dataframe
base_names = em_02.select('product_name').to_series().to_list()
base_prices = em_02.select('sale_price').to_series().to_list()

# 1. Identify which product names are new and need to be embedded.
names_to_embed = [name for name in base_names if name not in embedding_cache]

# 2. If there are any new names, process them in manageable batches.
if names_to_embed:
    total_batches = -(-len(names_to_embed) // BATCH_SIZE)  # Ceiling division to calculate total batches
    print(f"Found {len(names_to_embed)} new products to embed. Processing in {total_batches} batches of up to {BATCH_SIZE} items each...")
    
    # Iterate through the new names in chunks of BATCH_SIZE
    for i in range(0, len(names_to_embed), BATCH_SIZE):
        current_batch_num = (i // BATCH_SIZE) + 1
        
        # Define the current batch of names
        batch_names = names_to_embed[i:i + BATCH_SIZE]
        print(f"  - Processing batch {current_batch_num}/{total_batches} ({len(batch_names)} items)...")
        
        # Generate embeddings for the current batch
        batch_embeddings = generate_embeddings(batch_names)
        
        # 3. Update the cache with the newly generated embeddings from this batch
        for name, embedding in zip(batch_names, batch_embeddings):
            embedding_cache[name] = embedding

        # 4. Save the updated cache to disk after each batch. This adds robustness.
        #    If the script fails, progress from completed batches is not lost.
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
        print(f"  - Batch {current_batch_num} processed and cache saved.")
            
    print("All new embeddings have been generated and cached.")

# 5. Retrieve all embeddings from the cache to ensure the list is complete and in the correct order.
base_embeddings = [embedding_cache[name] for name in base_names]

data_for_df = {
    'product_name': base_names,
    'embedding': base_embeddings,
    'sale_price': base_prices
}
product_embeddings_df = pl.DataFrame(data_for_df)
product_embeddings_df.head()


# ## Find Best Match
import google.generativeai as genai
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
gemini_client = genai.Client()


# --- 1. CORE MATCHING LOGIC ---
def find_best_match_in_context(query_name, context_embeddings_np):
    """
    Finds the single most similar product in the context catalog for a given query name.
    """
    # The embedding for the query product is generated here, on-the-fly.
    query_embedding = embedding_from_string(query_name)
    similarities = np.dot(context_embeddings_np, query_embedding) / \
                   (np.linalg.norm(context_embeddings_np, axis=1) * np.linalg.norm(query_embedding))
    best_index = np.argmax(similarities)
    return best_index, similarities[best_index]


# --- 2. NEW GEMINI-BASED VERIFICATION LOGIC ---
def verify_matches_with_gemini(product_pairs: list[dict]):
    """
    Sends a list of product pairs to the Gemini API for verification.

    Args:
        product_pairs (list[dict]): A list of dictionaries, where each dict contains
                                    'query_product_name' and 'candidate_product_name'.

    Returns:
        list: A list of verification results from the LLM.
    """
    if not product_pairs:
        return []

    # Construct the prompt with up to 10 pairs
    prompt_lines = [
    """
    You are an expert product matching AI. Your primary function is to accurately determine if two product descriptions refer to the exact same underlying product.

    **Task:**
    You will be given pairs of product names. For each pair, you must decide if they are the same product.

    **Matching Criteria:**
    - **Core Product Identity:** The core product name, brand, and essential features (like model number or volume/size) must match.
    - **Ignore Superficial Differences:** Disregard noise like:
        - Advertiser names (e.g., "Advertiser: BestDeals")
        - Promotions and discounts (e.g., "Promotion: Save $100", "ลด 50%")
        - Minor descriptive text or marketing language (e.g., "(Visibly Reduces Wrinkles)")
        - Language differences (e.g., Thai mixed with English).
    - **Handle Variations Carefully:**
        - **Consider these as DIFFERENT products:** Different fundamental versions like 'Day Cream' vs. 'Night Cream', different sizes (e.g., '50ml' vs '100ml'), or different models (e.g., 'iPhone 13' vs 'iPhone 13 Pro').
        - **Consider these as the SAME product:** Only minor packaging variations or promotional text differences.

    **Input Format:**
    A list of product pairs will be provided.
    {product_pairs}

    **Output Format:**
    Your response MUST be a single valid JSON object. This object will contain one key, "verifications", which is a list of objects. Each object in the list corresponds to a product pair from the input and must contain a single key "match_found" with a boolean value (true or false). The order of your responses must exactly match the input order.

    **Example:**

    Input:
    ```
    1. Base Product: Apple iPhone 13 (Advertiser: BestDeals)
       Candidate Product: Apple iPhone 13
    2. Base Product: GlowUp Advanced Day Cream 50ml (For daily use)
       Candidate Product: GlowUp Advanced Night Cream 50ml (For nighttime repair)
    3. Base Product: ยาสีฟันคอลเกต 100g
       Candidate Product: Colgate Toothpaste 100g
    ```

    Output:
    ```json
    {
      "verifications": [
        {
          "match_found": true
        },
        {
          "match_found": false
        },
        {
          "match_found": true
        }
      ]
    }
    """
]

    for i, pair in enumerate(product_pairs, 1):
        prompt_lines.append(f"\n{i}. Base Product: '{pair['query_product_name']}'")
        prompt_lines.append(f"   Candidate Product: '{pair['candidate_product_name']}'")

    prompt = "\n".join(prompt_lines)

    try:
        # 3. Call the generate_content method with the prompt.
        print(f"\nSending {len(product_pairs)} pairs to Gemini for verification...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=prompt,
            config={
                "response_mime_type":"application/json"
            }
        )
        
        # --- FIXED SECTION END ---
        
        # The response.text already contains the JSON string
        result_json = json.loads(response.text)
        return result_json.get('verifications', [])

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        # Return a list of 'False' results to prevent crashing the main loop
        return [{'match_found': False} for _ in product_pairs]

# --- 3. MODIFIED MAIN PROCESSING SCRIPT (Unchanged) ---

# Define the batch size for each API call
API_CALL_SIZE = 10

# df_query corresponds to the new products to be checked.
# df_base corresponds to the existing product catalog.
df_query = em_01
df_base = product_embeddings_df
base_embeddings_np = np.array(df_base['embedding'].to_list())

results_list = []
query_rows = list(df_query.iter_rows(named=True))

print(f"Starting product matching in chunks of {API_CALL_SIZE}...")

# Process the query dataframe in sequential chunks
for i in range(0, len(query_rows), API_CALL_SIZE):
    product_chunk = query_rows[i:i + API_CALL_SIZE]
    chunk_num = (i // API_CALL_SIZE) + 1
    print(f"\n--- Processing Chunk {chunk_num} ({len(product_chunk)} products) ---")
    
    chunk_data_for_verification = []
    chunk_original_data = []

    for row in product_chunk:
        query_product_name = row.get('product_name')
        if not query_product_name:
            continue

        # Find the best candidate match from the base catalog
        match_index, similarity = find_best_match_in_context(query_product_name, base_embeddings_np)
        matched_product_info = df_base.row(match_index, named=True)
        
        if similarity > 0.8:
            chunk_data_for_verification.append({
                'query_product_name': query_product_name,
                'candidate_product_name': matched_product_info['product_name']
            })
            
            chunk_original_data.append({
                'original_row': row,
                'match_info': matched_product_info,
                'similarity': similarity
            })

    # Call the Gemini API for the entire chunk
    llm_results = verify_matches_with_gemini(chunk_data_for_verification)

    if len(llm_results) != len(chunk_original_data):
        print(f"Warning: Mismatch between sent items ({len(chunk_original_data)}) and received results ({len(llm_results)}). Skipping chunk.")
        continue

    for original_data, llm_result in zip(chunk_original_data, llm_results):
        try:
            is_match = llm_result.get('match_found', False)
        except (AttributeError, TypeError):
            print(f"Could not parse LLM result: {llm_result}. Treating as no match.")
            is_match = False

        query_row = original_data['original_row']
        match_info = original_data['match_info']
        similarity_score = original_data['similarity']

        results_list.append({
            'base_product_name (from line)': query_row['product_name'],
            'base_product_name (from watson)': match_info['product_name'],
            'price_from_line': query_row['sale_price'],
            'price_from_watson': match_info['sale_price'],
            'similarity': float(similarity_score),
            'verified_match': is_match
        })
    
    print(f"Finished processing Chunk {chunk_num}.")


print("\n--- All chunks processed. Final results compiled. ---")

if results_list:
    results_df = pl.DataFrame(results_list)

    # Save Results to BigQuery
    print("\n--- Saving results to BigQuery... ---")
    try:
        # Convert Polars DataFrame to Ibis in-memory table expression
        ibis_table_expr = ibis.memtable(results_df)

        dataset_name = "ecommerce_product"
        table_name_only = "product_comparison_price"
        full_table_name = f"{dataset_name}.{table_name_only}"

        # FIX: Use con.list_tables() to check for existence.
        # This lists all tables in the specified dataset and checks for membership.
        if table_name_only in con.list_tables(database=dataset_name):
            # If it exists, append data.
            print(f"Table '{full_table_name}' already exists. Appending data...")
            con.insert(
                table_name_only,
                ibis_table_expr,
                database=dataset_name,
                overwrite=False  # This ensures we append
            )
            print(f"Results appended to existing BigQuery table '{full_table_name}'.")
        else:
            # If it does not exist, create it.
            print(f"Table '{full_table_name}' does not exist. Creating table...")
            con.create_table(
                table_name_only,
                ibis_table_expr,
                database=dataset_name
            )
            print(f"Table '{full_table_name}' created and results saved to BigQuery.")

    except Exception as e:
        # This will now catch genuine errors during creation or insertion.
        print(f"An unexpected error occurred while saving results to BigQuery: {e}")

else:
    print("No matches were found or verified, so no data to save.")
