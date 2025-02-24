import os
import pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API Key from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ensure API key is set
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Please set PINECONE_API_KEY in your .env file.")

def create_pinecone_index(index_name=PINECONE_INDEX_NAME):
    """
    Check if the Pinecone index exists and create it if necessary.
    Uses serverless spec with AWS configuration.
    """
    try:
        # Initialize Pinecone Client
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if the index exists
        if not pc.has_index(index_name):
            print(f"Index {index_name} not found. Creating...")
            pc.create_index(
                name=index_name,
                dimension=3584,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index {index_name} created successfully.")
        else:
            print(f"Index {index_name} already exists.")
            
        return pc.Index(index_name)
            
    except Exception as e:
        print(f"Error creating/accessing Pinecone index: {str(e)}")
        raise

if __name__ == "__main__":
    # Create index when script is run directly
    create_pinecone_index()