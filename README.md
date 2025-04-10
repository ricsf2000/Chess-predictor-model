# Chess-predictor-model
Chess game outcome predictor model made for the Data Science II final project.

## Setup Instructions

1. **Download the Chess Dataset**:
   - Download the dataset from: https://uofh-my.sharepoint.com/:u:/g/personal/ramoral4_cougarnet_uh_edu/EXBjcNdmlglInGvllzDtAu0BD-d-fhdlXgf-vIPS0EFbDQ?e=21zRXA 


3. **Verify Dataset Integrity**:
   - Verify that the processed dataset matches our reference hash:
     ```
     python verify_dataset.py
     ```
   - The hash reference file is included in the repository: `lichess_processed_1000000_games_first_15_moves.sha256`