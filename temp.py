import re
import pinecone 
from pinecone import ServerlessSpec
import time

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key="pcsk_2yYsZE_KG5KrrY9CWrP694FTCnsLMatAVXrynUShvqn8hsRQKSMpzgj6TXE7JVHY5g287r")

# Wait for index to be ready
while not pc.describe_index("openwebui-experimental").status['ready']:
    print("Waiting for Pinecone index to be ready...")
    time.sleep(1)

# Connect to the index
index = pc.Index("openwebui-experimental")
print(f"Connected to Pinecone index: openwebui-experimental")

# Dictionary containing PowerPoint filenames and their respective new links
pptx_links = {
    "FULL_HawkPartners Astellas Oncology CX Tracker_Wave 6 +LC 20240426.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=9fb66a37-15f2-47a6-87ed-618b859fbce4&cid=35f815faa57423bb",
    "HCP Access Perceptions Update 2024 Quant Final_Report_13JAN2025.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=600c9f93-eec6-43c0-9510-9614bcc5dab6&cid=35f815faa57423bb",
    "Leqvio Demand Study Findings_March 31 2023.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=41d5d512-a2c7-4cbd-a829-b3e32a76105b&cid=35f815faa57423bb",
    "Opportunity for RTD and RTU Antibiotics_HawkPartners Abridged Presentation 011625.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=755498f6-d98d-4cb5-92d0-7bd28a4bcdd8&cid=35f815faa57423bb",
    "UCB Caregiver DS and LGS W5 Full Report Template .pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=5642d90f-7f04-4e18-9877-a97ed9a94f2b&cid=35f815faa57423bb",
    "UCB Dravet and LGS Caregiver ATU Wave 4 Report_Sept 2024_revised.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=0c527f93-d4c6-4067-a4f1-8b3fecd54664&cid=35f815faa57423bb",
    "US HCP LBCL Segmentation Final Report_4OCT2024.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=fb4196f7-23bd-4f0e-8511-1a6c671d7355&cid=35f815faa57423bb",
    "US mUC A39 Message Refinement – Final Report 15SEP2024.pptx": "https://onedrive.live.com/personal/35f815faa57423bb/_layouts/15/doc.aspx?resid=abe2b0f5-2121-4521-bdfe-9fa5eb9ed9e3&cid=35f815faa57423bb",
}

# Use the namespace parameter
namespace = "ns1"

# Statistics counters
total_ids_checked = 0
total_ids_found = 0
total_ids_updated = 0

# Process each PowerPoint file in the dictionary
for pptx_name, new_link in pptx_links.items():
    print(f"\nProcessing: {pptx_name}")
    
    # Replace non-ASCII characters with ASCII equivalents
    ascii_pptx_name = pptx_name.replace("–", "-")
    
    # Track stats for this file
    file_ids_checked = 0
    file_ids_found = 0
    file_ids_updated = 0
    
    # Build a list of variations to check (base filename and _1 through _175)
    variations_to_check = [ascii_pptx_name]  # Base filename
    for i in range(1, 176):  # Variations _1 through _175
        variations_to_check.append(f"{ascii_pptx_name}_{i}")
    
    # Process variations in batches to avoid request size limitations
    batch_size = 50  # Adjust if needed
    
    for batch_start in range(0, len(variations_to_check), batch_size):
        batch_end = min(batch_start + batch_size, len(variations_to_check))
        batch = variations_to_check[batch_start:batch_end]
        file_ids_checked += len(batch)
        
        try:
            # Fetch this batch of IDs
            response = index.fetch(ids=batch, namespace=namespace)
            
            # Process any found IDs
            if hasattr(response, 'vectors') and response.vectors:
                found_ids = list(response.vectors.keys())
                file_ids_found += len(found_ids)
                
                print(f"  Found {len(found_ids)} IDs in batch {batch_start//batch_size + 1}")
                
                # Update each found ID
                for found_id in found_ids:
                    try:
                        index.update(id=found_id, set_metadata={"one_drive_link": new_link}, namespace=namespace)
                        file_ids_updated += 1
                    except Exception as e:
                        print(f"  Error updating {found_id}: {e}")
            
        except Exception as e:
            print(f"  Error processing batch {batch_start//batch_size + 1}: {e}")
    
    # Update totals
    total_ids_checked += file_ids_checked
    total_ids_found += file_ids_found
    total_ids_updated += file_ids_updated
    
    # Print summary for this file
    print(f"  Summary for {pptx_name}:")
    print(f"  - Checked {file_ids_checked} potential IDs")
    print(f"  - Found {file_ids_found} IDs")
    print(f"  - Updated {file_ids_updated} IDs")

# Print overall summary
print("\nOverall Summary:")
print(f"Total IDs checked: {total_ids_checked}")
print(f"Total IDs found: {total_ids_found}")
print(f"Total IDs updated: {total_ids_updated}")
print("\nUpdate complete.")
