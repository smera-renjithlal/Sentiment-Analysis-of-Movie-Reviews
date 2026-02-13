# Import required libraries 
import pandas as pd 
import numpy as np
import re 
from sklearn.model_selection import train_test_split
# Load the IMDB dataset print("Loading IMDB   Dataset...") 
df = pd.read_csv('IMDB_Dataset.csv')
# Display initial dataset information 
print(f"Initial Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}") 
print(f"\nFirst few rows:") 
print(df.head()) 
print(f"\nDataset Info:")
print(df.info()) 
print(f"\nClass Distribution:") 
print(df['sentiment'].value_counts())

# Check for missing values print("\n" + "="*50) 
print("DATA QUALITY ASSESSMENT") 
print("="*50) 
print(f"\nMissing Values:") 
print(df.isnull().sum())
# Check for duplicate records 
print(f"\nDuplicate Records: {df.duplicated().sum()}") 
print(f"Duplicate Reviews (based on text): {df.duplicated(subset=['review']).sum()}")
# Display sample noisy data
print(f"\nSample Review (showing noise):")
print(df['review'].iloc[0][:500])  # First 500 characters


# ============================================
# DATA CLEANING - TECHNIQUE 1 
# ============================================
def clean_text(text):     
 """  Clean text data by removing noise and normalizing         
        Steps:
1.	Convert to lowercase
2.	Remove HTML tags
3.	Remove URLs
4.	Remove special characters and numbers
5.	Remove extra whitespace    """    
 # Convert to lowercase     
 text = text.lower()         
 # Remove HTML tags     
 text = re.sub(r'<.*?>', '', text)         
 # Remove URLs     
 text = re.sub(r'http\S+|www\S+', '', text)     
 # Remove special characters and numbers, keep only letters and spaces     
 text = re.sub(r'[^a-z\s]', ' ', text)         
 # Remove extra whitespace
 text = re.sub(r'\s+', ' ', text).strip()         
 return text
# Apply text cleaning 
print("\n" + "="*50) 
print("APPLYING DATA CLEANING") 
print("="*50) 
print("\nCleaning text data...") 
df['cleaned_review'] = df['review'].apply(clean_text)
# Display before and after cleaning 
print("\nBefore Cleaning:")
print(df['review'].iloc[0][:300]) 
print("\nAfter Cleaning:") 
print(df['cleaned_review'].iloc[0][:300])
# Remove duplicate records 
print(f"\nRemoving duplicate records...") 
initial_size = len(df) 
df = df.drop_duplicates(subset=['cleaned_review'], keep='first') 
final_size = len(df) 
print(f"Records removed: {initial_size - final_size}") 
print(f"Dataset shape after removing duplicates: {df.shape}")
# Reset index after removing duplicates 
df = df.reset_index(drop=True)


# ============================================
# DATA REDUCTION - TECHNIQUE 2: STRATIFIED SAMPLING # ============================================
def stratified_sampling(dataframe, sample_size, class_column):    
    """
    Perform stratified sampling to maintain class distribution
        Parameters:
-	dataframe: Input DataFrame
-	sample_size: Total number of samples to draw
-	class_column: Column name containing class labels
        Returns:
-	Sampled DataFrame with preserved class distribution
    """     
    # Calculate samples per class (proportional sampling)    
    class_counts = dataframe[class_column].value_counts()    
    class_proportions = class_counts / len(dataframe)         
    print(f"Original class distribution:")     
    print(class_counts)     
    print(f"\nClass proportions:")     
    print(class_proportions)         
    # Sample from each class proportionally    
    sampled_dfs = []     
    for class_label, proportion in class_proportions.items():         
        class_sample_size = int(sample_size * proportion)         
        class_df = dataframe[dataframe[class_column] == class_label]         
        class_sample = class_df.sample(n=class_sample_size, random_state=42)        
        sampled_dfs.append(class_sample)         

    # Combine samples from all classes     
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)         

    # Shuffle the combined sample     
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)         
    return sampled_df
# Apply stratified sampling print("\n" + "="*50) 
print("APPLYING DATA REDUCTION (STRATIFIED SAMPLING)") 
print("="*50)
# Define sample size (e.g., 20% of original data) 
sample_percentage = 0.2 
sample_size = int(len(df) * sample_percentage)
print(f"\nOriginal dataset size: {len(df)}")
print(f"Sample size ({sample_percentage*100}%): {sample_size}")
# Perform stratified sampling 
df_sampled = stratified_sampling(df, sample_size, 'sentiment')
print(f"\nSampled dataset shape: {df_sampled.shape}")
print(f"\nSampled class distribution:")
print(df_sampled['sentiment'].value_counts())
# Verify proportions are maintained 
print(f"\nOriginal proportions:") 
print(df['sentiment'].value_counts(normalize=True)) 
print(f"\nSampled proportions:")
print(df_sampled['sentiment'].value_counts(normalize=True))


# Save cleaned full dataset 
print("\n" + "="*50) 
print("SAVING PREPROCESSED DATA") 
print("="*50) 
df.to_csv('IMDB_cleaned_full.csv', index=False)
print("\nCleaned full dataset saved as 'IMDB_cleaned_full.csv'")
# Save sampled dataset df_sampled.to_csv('IMDB_cleaned_sampled.csv', index=False) print("Sampled dataset saved as 'IMDB_cleaned_sampled.csv'")
# Summary statistics print("\n" + "="*50) print("PREPROCESSING SUMMARY") 
print("="*50) 
print(f"Original dataset size: 50,000 records") 
print(f"After duplicate removal: {len(df)} records") 
print(f"Sampled dataset size: {len(df_sampled)} records") 
print(f"Data reduction: {(1 - len(df_sampled)/len(df))*100:.2f}%") 
print("\nPreprocessing completed successfully!")
