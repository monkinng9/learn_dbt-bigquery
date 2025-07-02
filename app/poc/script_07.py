# # Import Necessary Library
from uu import Error
import ibis
import ibis.expr.datatypes as dt
from google.oauth2 import service_account

# # Load credentials from a service account JSON file
credentials = service_account.Credentials.from_service_account_file(
    '/opt/spark/work-dir/google-service-account.json',
    scopes=['https://www.googleapis.com/auth/bigquery'] # Specify necessary scopes
)

# # Create BigQuery Session with Ibis

con = ibis.bigquery.connect(
    project_id='personal-use-461616',
    credentials=credentials,
    location="asia-southeast1"
)

raw_df = con.table("ecommerce_product.product_price_raw_data")
raw_df


# # Transform Data

# ## Split Data by Website
em_01_df = raw_df.filter(raw_df.ecommerce_name == 'ecommerce_01')
em_02_df = raw_df.filter(raw_df.ecommerce_name == 'ecommerce_02')

# # Embedding Base Product Name
from dotenv import load_dotenv
# Load environment variables from the specified path
load_dotenv(dotenv_path='/home/sparky/.dbt/.env')

from openai import OpenAI
import pickle
import numpy as np
import json
import os
import polars as pl
import math

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

# ## Embed base product name
# Define a dynamic batch size. You can adjust this value based on API limits and performance.
BATCH_SIZE = 100

print("Preparing base by generating embeddings for all base products...")
# Get all product names from the base dataframe
filtered_data_pl = em_02_df.select('product_name_sha256', 'product_name', 'sale_price', 'ingest_timestamp_utc').to_polars()

base_names_list = filtered_data_pl['product_name'].to_list() # Convert to list for batching with OpenAI API

total_batches = math.ceil(len(base_names_list) / BATCH_SIZE)
print(f"Found {len(base_names_list)} new products to embed. Processing in {total_batches} batches of up to {BATCH_SIZE} items each...")

all_embeddings = [] # Collect all embeddings here

for i in range(0, len(base_names_list), BATCH_SIZE):
    batch_names = base_names_list[i:i + BATCH_SIZE]
    
    # Generate embeddings for the current batch
    new_embeddings = generate_embeddings(batch_names)
    
    all_embeddings.extend(new_embeddings) # Add embeddings in order

    print(f"Finished processing batch {i // BATCH_SIZE + 1} of {total_batches}...")

# Add embeddings to the Polars DataFrame
filtered_data_pl = filtered_data_pl.with_columns(pl.Series(name="embedding", values=all_embeddings))

print("Base product embeddings generated and cached successfully.")

product_embeddings_df = filtered_data_pl.with_columns(
    pl.Series(name="embedding", values=all_embeddings)
)

# ## Find Best Match & Verify with Gemini
from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
gemini_client = genai.Client()

# --- 1. CORE MATCHING LOGIC (REVISED) ---
def find_best_match(query_embedding, context_embeddings_np):
    """
    Finds the single most similar product in the context catalog for a given query embedding.

    Args:
        query_embedding (np.ndarray): The embedding of the query product.
        context_embeddings_np (np.ndarray): NumPy array of embeddings for the context products.

    Returns:
        tuple: A tuple containing the index of the best match and its similarity score.
    """
    # Calculate cosine similarity
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
    A list of product pairs will be provided below.

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

    # --- DEBUG: Print the exact prompt being sent ---
    print("\n--- PROMPT SENT TO GEMINI ---")
    print(prompt)
    print("-----------------------------\n")

    try:
        # 3. Call the generate_content method with the prompt.
        print(f"\nSending {len(product_pairs)} pairs to Gemini for verification...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # --- DEBUG: Print the raw response ---
        print("\n--- RAW RESPONSE FROM GEMINI ---")
        print(response.text)
        print("--------------------------------\n")

        # The response.text already contains the JSON string
        result_json = json.loads(response.text)
        return result_json.get('verifications', [])

    except Exception as e:
        print(f"An error occurred while calling the Gemini API or parsing its response: {e}")
        # Return a list of 'False' results to prevent crashing the main loop
        return [{'match_found': False} for _ in product_pairs]

# --- 3. MODIFIED MAIN PROCESSING SCRIPT ---

# Define the batch size for each API call
API_CALL_SIZE = 10
import polars as pl
import numpy as np
import ibis

# Assume df_query, df_base, generate_embeddings, find_best_match, 
# verify_matches_with_gemini, and the ibis connection 'con' are already defined.

# df_query corresponds to the new products to be checked.
# df_base corresponds to the existing product catalog.
df_query = em_01_df.select('product_name_sha256', 'product_name', 'sale_price', 'ingest_timestamp_utc').to_polars()
df_base = product_embeddings_df # This already has embeddings
base_embeddings_np = df_base.select(
    pl.col('embedding').list.to_struct()
).unnest('embedding').to_numpy()

# MODIFICATION: We will collect result DataFrames from each chunk in a list.
all_results_dfs = []

print(f"Starting product matching in chunks of {API_CALL_SIZE}...")

# MODIFICATION: Iterate over slices of the DataFrame instead of converting to a list of rows first.
# This is more memory-efficient for large query dataframes.
for chunk_num, product_chunk_df in enumerate(df_query.iter_slices(n_rows=API_CALL_SIZE), 1):
    print(f"\n--- Processing Chunk {chunk_num} ({product_chunk_df.height} products) ---")
    
    # Get names and generate embeddings for the current chunk
    chunk_names = product_chunk_df['product_name'].to_list()
    chunk_embeddings = generate_embeddings(chunk_names)

    chunk_data_for_verification = []
    chunk_original_data = []

    # Iterate over the rows of the chunk DataFrame to find matches
    for idx, query_row in enumerate(product_chunk_df.iter_rows(named=True)):
        query_product_name = query_row.get('product_name')
        query_embedding = chunk_embeddings[idx]

        if not query_product_name:
            continue

        # Find the best candidate match from the base catalog
        match_index, similarity = find_best_match(query_embedding, base_embeddings_np)
        
        # Set a threshold for similarity before sending to the LLM
        if similarity > 0.8:
            matched_product_info = df_base.row(match_index, named=True)
            
            chunk_data_for_verification.append({
                'query_product_name': query_product_name,
                'candidate_product_name': matched_product_info['product_name']
            })
            
            chunk_original_data.append({
                'original_row': query_row,
                'match_info': matched_product_info,
                'similarity': similarity
            })

    # Call the Gemini API for the entire chunk if there's anything to verify
    if not chunk_data_for_verification:
        print("No potential matches found in this chunk. Skipping Gemini call.")
        continue

    llm_results = verify_matches_with_gemini(chunk_data_for_verification)

    if len(llm_results) != len(chunk_original_data):
        print(f"Warning: Mismatch between sent items ({len(chunk_original_data)}) and received results ({len(llm_results)}). Skipping chunk.")
        continue

    # Process results from LLM
    # MODIFICATION: Create a list of dictionaries for the current chunk's results.
    chunk_results_list = []
    for original_data, llm_result in zip(chunk_original_data, llm_results):
        try:
            is_match = llm_result.get('match_found', False)
        except (AttributeError, TypeError):
            print(f"Could not parse LLM result: {llm_result}. Treating as no match.")
            is_match = False

        query_row = original_data['original_row']
        match_info = original_data['match_info']
        similarity_score = original_data['similarity']

        chunk_results_list.append({
            'query_product_name': query_row['product_name'],
            'matched_product_name': match_info['product_name'],
            'query_price': query_row['sale_price'],
            'matched_price': match_info['sale_price'],
            'similarity': float(similarity_score),
            'verified_match': is_match,
            'ingest_timestamp_utc': query_row['ingest_timestamp_utc']
        })
    
    # MODIFICATION: If there are results, convert the list to a DataFrame and append to our list of DataFrames.
    if chunk_results_list:
        chunk_results_df = pl.DataFrame(chunk_results_list)
        all_results_dfs.append(chunk_results_df)

    print(f"Finished processing Chunk {chunk_num}.")


print("\n--- All chunks processed. Final results compiled. ---")

# MODIFICATION: Concatenate all the chunk DataFrames into one final DataFrame.
if all_results_dfs:
    results_df = pl.concat(all_results_dfs)

    # Add partition columns
    results_df = results_df.with_columns(
        pl.col("ingest_timestamp_utc").dt.date().alias("ingest_date")
    ).rename({
        "query_product_name": "em01_product_name",
        "matched_product_name": "em02_product_name",
        "query_price": "em01_price",
        "matched_price": "em02_price",
    })

    # Save Results to BigQuery
    print("\n--- Saving results to BigQuery... ---")
    try:
        # Convert Polars DataFrame to Ibis in-memory table expression
        ibis_table_expr = ibis.memtable(results_df)
        # Add a date column for partitioning
        ibis_table_expr = ibis_table_expr.mutate(
            ingest_date=ibis_table_expr.ingest_timestamp_utc.cast(dt.date))

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
                database=dataset_name,
                partition_by=['ingest_date']
            )


            con.raw_sql(search_index_sql).execute()

    except Exception as e:
        # This will now catch genuine errors during creation or insertion.
        print(f"An unexpected error occurred while saving results to BigQuery: {e}")
        raise Error

else:
    print("No matches were found or verified, so no data to save.")
