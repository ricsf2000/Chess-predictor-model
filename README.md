# Chess-predictor-model
Chess game outcome predictor model made for the Data Science II final project.

## Setup Instructions

1. **Download the Chess Dataset**:
   - Download the January 2025 standard rated games from the Lichess database: https://database.lichess.org/
   - Save the file as `lichess_db_standard_rated_2025-01.pgn.zst` in the parent directory

2. **Process the Dataset**:
   - Run the preprocessing script to extract features from the first 1 million games:
     ```
     python preprocess_games.py
     ```
   - This will create a processed dataset file in the data directory

3. **Verify Dataset Integrity**:
   - Verify that the processed dataset matches our reference hash:
     ```
     python verify_dataset.py
     ```
   - The hash reference file is included in the repository: `lichess_processed_1000000_games_first_15_moves.sha256`