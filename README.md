# Chess-predictor-model
Chess game outcome predictor model made for the Data Science II final project.

## Setup Instructions

1. **Chess Dataset**:
   - The dataset will be available as a .zip file called data.zip

3. **Verify Dataset Integrity**:
   - Verify that the processed dataset matches our reference hash:
     ```
     python verify_dataset.py
     ```
   - The hash reference file is included in the repository: `lichess_processed_1000000_games_first_15_moves.sha256`
   - Ensure that the following pkl file is in the same directory:
   `lichess_processed_1000000_games_first_15_moves.pkl`