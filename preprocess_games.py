import os
import hashlib
import zstandard as zstd
import chess.pgn
import io
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import re

def preprocess_games(input_file, output_dir="./data", max_games=1_000_000, first_n_moves=15):
    """
    Preprocess chess games from a PGN file, keeping only essential data
    
    Args:
    - input_file: Path to the compressed PGN file (.pgn.zst)
    - output_dir: Directory to save the processed data
    - max_games: Maximum number of games to process
    - first_n_moves: Only keep this many moves per game
    
    Returns:
    - Path to the processed games file
    - Dictionary of stats
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    processed_path = os.path.join(output_dir, f"lichess_processed_{max_games}_games_first_{first_n_moves}_moves.pkl")
    stats_path = os.path.join(output_dir, f"lichess_stats_{max_games}_games.json")
    
    # Check if processed file already exists
    if os.path.exists(processed_path) and os.path.exists(stats_path):
        print(f"Processed games file already exists at {processed_path}")
        
        # Load stats file
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            
        print("\n--- Game Outcome Analysis ---")
        print(f"Total Games: {stats['total_games']}")
        print("\nGame Outcome Counts:")
        for outcome, count in stats['outcomes'].items():
            print(f"{outcome}: {count}")
        
        print("\nGame Outcome Percentages:")
        for outcome, percentage in stats['percentages'].items():
            print(f"{outcome}: {percentage:.2f}%")
            
        return processed_path, stats
    
    # Regular expression to extract evaluation from comments
    eval_pattern = re.compile(r'\[%eval\s+([-+]?\d+\.\d+|\#-?\d+)\]')
    
    # Store processed games
    games = []
    
    # Outcome tracking
    outcomes = {
        'White Wins': 0,
        'Black Wins': 0,
        'Draws': 0
    }
    
    # Player rating tracking
    rating_data = {
        'white': [],
        'black': [],
        'total': []
    }
    
    # ECO code tracking
    eco_codes = {}
    
    # Total games counter
    total_games = 0
    
    # Open zstd compressed file
    print(f"Processing first {max_games} games from {input_file}...")
    with open(input_file, 'rb') as file:
        # Create decompressor
        dctx = zstd.ZstdDecompressor()
        
        # Stream decompressed data
        with dctx.stream_reader(file) as reader:
            # Create text stream
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            
            # Create progress bar
            pbar = tqdm(total=max_games, desc="Processing games")
            
            while total_games < max_games:
                # Read game
                game = chess.pgn.read_game(text_reader)
                
                # Break if no more games
                if game is None:
                    break
                
                # Increment total games
                total_games += 1
                pbar.update(1)
                
                # Get the result
                result = game.headers.get('Result', '')
                
                # Skip incomplete games
                if result not in ['1-0', '0-1', '1/2-1/2']:
                    continue
                
                # Extract only essential game data
                game_data = {
                    # Essential data for prediction
                    'result': result,  # '1-0' (white wins), '0-1' (black wins), '1/2-1/2' (draw)
                    'result_class': 0 if result == '1-0' else (1 if result == '0-1' else 2),  # 0=white win, 1=black win, 2=draw
                    'white_elo': int(game.headers.get('WhiteElo', 0)),
                    'black_elo': int(game.headers.get('BlackElo', 0)),
                    'elo_diff': int(game.headers.get('WhiteElo', 0)) - int(game.headers.get('BlackElo', 0)),
                    'eco': game.headers.get('ECO', ''),  # Opening classification code
                    'time_control': game.headers.get('TimeControl', ''),
                }
                
                # Extract time control information
                if game_data['time_control']:
                    try:
                        tc_parts = game_data['time_control'].split('+')
                        base_time = int(tc_parts[0])
                        increment = int(tc_parts[1]) if len(tc_parts) > 1 else 0
                        
                        game_data['base_time_seconds'] = base_time
                        game_data['increment_seconds'] = increment
                        
                        # Game time classification
                        if base_time < 180:
                            game_data['time_class'] = 'bullet'
                        elif 180 <= base_time < 600:
                            game_data['time_class'] = 'blitz'
                        elif 600 <= base_time < 1800:
                            game_data['time_class'] = 'rapid'
                        else:
                            game_data['time_class'] = 'classical'
                    except (ValueError, IndexError):
                        pass
                
                # Process the moves - only keep the first n moves
                moves = []
                evals = []
                legal_moves_count = []
                board = chess.Board()
                
                # Extract moves and their features
                for i, node in enumerate(game.mainline()):
                    # Stop after first_n_moves
                    if i >= first_n_moves * 2:  # *2 because each move has white and black
                        break
                    
                    move = node.move
                    uci_move = move.uci()
                    moves.append(uci_move)
                    
                    # Count legal moves before playing the move (mobility)
                    legal_moves_count.append(board.legal_moves.count())
                    
                    # Extract evaluation if available
                    comment = node.comment
                    if comment:
                        eval_match = eval_pattern.search(comment)
                        if eval_match:
                            eval_str = eval_match.group(1)
                            if eval_str.startswith('#'):
                                # Handle mate scores (convert to large numerical values)
                                mate_moves = int(eval_str[1:])
                                eval_value = 100 if mate_moves > 0 else -100
                            else:
                                eval_value = float(eval_str)
                            evals.append(eval_value)
                    
                    # Apply the move to the board
                    board.push(move)
                
                # Skip games with too few moves
                if len(moves) < first_n_moves:
                    continue
                
                # Store the first n moves only
                game_data['moves'] = moves[:first_n_moves*2]  # first n full moves (white and black)
                
                # Store evaluations if available
                if evals:
                    game_data['evals'] = evals[:first_n_moves*2]
                    
                    # Add final position evaluation as a feature
                    if evals:
                        game_data['final_eval'] = evals[-1]
                
                # Add move counts as a feature
                game_data['legal_moves_count'] = legal_moves_count[:first_n_moves*2]
                
                # Calculate material balance after n moves
                white_material = 0
                black_material = 0
                
                piece_values = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9
                }
                
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        value = piece_values.get(piece.piece_type, 0)
                        if piece.color == chess.WHITE:
                            white_material += value
                        else:
                            black_material += value
                
                game_data['white_material'] = white_material
                game_data['black_material'] = black_material
                game_data['material_balance'] = white_material - black_material
                
                # Add castling information
                game_data['white_can_castle'] = board.has_castling_rights(chess.WHITE)
                game_data['black_can_castle'] = board.has_castling_rights(chess.BLACK)
                
                # Simplified features from board state
                # Center control (d4, d5, e4, e5)
                center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
                white_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
                black_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
                game_data['white_center_control'] = white_center_control
                game_data['black_center_control'] = black_center_control
                
                # Add to processed games
                games.append(game_data)
                
                # Update stats
                if result == '1-0':
                    outcomes['White Wins'] += 1
                elif result == '0-1':
                    outcomes['Black Wins'] += 1
                elif result == '1/2-1/2':
                    outcomes['Draws'] += 1
                
                # Player ratings
                white_elo = game_data['white_elo']
                black_elo = game_data['black_elo']
                if white_elo > 0:
                    rating_data['white'].append(white_elo)
                if black_elo > 0:
                    rating_data['black'].append(black_elo)
                if white_elo > 0 and black_elo > 0:
                    rating_data['total'].append((white_elo + black_elo) / 2)
                
                # ECO code
                eco = game_data['eco']
                if eco:
                    eco_codes[eco] = eco_codes.get(eco, 0) + 1
            
            pbar.close()
    
    print(f"Total games processed: {len(games)}")
    
    # Calculate percentages
    outcome_percentages = {
        key: (value / len(games) * 100) 
        for key, value in outcomes.items()
    }
    
    # Calculate rating statistics
    rating_stats = {
        'white': {
            'min': min(rating_data['white']) if rating_data['white'] else 0,
            'max': max(rating_data['white']) if rating_data['white'] else 0,
            'avg': sum(rating_data['white']) / len(rating_data['white']) if rating_data['white'] else 0
        },
        'black': {
            'min': min(rating_data['black']) if rating_data['black'] else 0,
            'max': max(rating_data['black']) if rating_data['black'] else 0,
            'avg': sum(rating_data['black']) / len(rating_data['black']) if rating_data['black'] else 0
        },
        'total': {
            'min': min(rating_data['total']) if rating_data['total'] else 0,
            'max': max(rating_data['total']) if rating_data['total'] else 0,
            'avg': sum(rating_data['total']) / len(rating_data['total']) if rating_data['total'] else 0
        }
    }
    
    # Get top ECO codes
    top_eco_codes = {k: v for k, v in sorted(eco_codes.items(), key=lambda item: item[1], reverse=True)[:10]}
    
    # Compile stats
    stats = {
        'total_games': len(games),
        'outcomes': outcomes,
        'percentages': outcome_percentages,
        'rating_stats': rating_stats,
        'top_eco_codes': top_eco_codes,
        'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save processed games
    print(f"Saving processed games to {processed_path}...")
    with open(processed_path, 'wb') as f:
        pickle.dump(games, f)
    
    # Save stats as JSON
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Processed games saved to {processed_path}")
    print(f"Stats saved to {stats_path}")
    
    # Note about hash verification
    print("\nNOTE: To verify dataset integrity, use the verify_dataset.py script with the expected hash.")
    
    return processed_path, stats

def main():
    """
    Main function to preprocess the Lichess dataset
    """
    # Set parameters
    INPUT_FILE = "../lichess_db_standard_rated_2025-01.pgn.zst"  # Use the provided path
    MAX_GAMES = 1_000_000
    FIRST_N_MOVES = 15
    OUTPUT_DIR = "./data"
    
    # Process dataset
    processed_path, stats = preprocess_games(INPUT_FILE, OUTPUT_DIR, MAX_GAMES, FIRST_N_MOVES)
    
    if processed_path and stats:
        # Print detailed dataset information
        print("\n=== DATASET INFORMATION ===")
        print(f"Dataset: Lichess Standard Rated Games (2025-01)")
        print(f"Samples: {stats['total_games']} games")
        print(f"Moves per game: {FIRST_N_MOVES}")
        print(f"Processed on: {stats['processed_date']}")
        
        print("\n--- Game Outcome Distribution ---")
        print(f"White Wins: {stats['percentages']['White Wins']:.2f}% ({stats['outcomes']['White Wins']} games)")
        print(f"Black Wins: {stats['percentages']['Black Wins']:.2f}% ({stats['outcomes']['Black Wins']} games)")
        print(f"Draws: {stats['percentages']['Draws']:.2f}% ({stats['outcomes']['Draws']} games)")
        
        print("\n--- Rating Information ---")
        print(f"Average Rating: {stats['rating_stats']['total']['avg']:.1f}")
        print(f"Rating Range: {stats['rating_stats']['total']['min']} - {stats['rating_stats']['total']['max']}")
        
        print("\n--- Top ECO Codes ---")
        for eco, count in stats['top_eco_codes'].items():
            print(f"{eco}: {count} games ({count/stats['total_games']*100:.1f}%)")
        
        print("\n--- Dataset Description for Research Paper ---")
        description = f"""Our dataset is composed of {stats['total_games']:,} standard rated games played on lichess.org in 01/2025. The dataset was collected from the Lichess database (https://database.lichess.org/). Our classes are composed of the three game outcomes that can occur in chess: White Wins, Black Wins, and Draw. The distribution of our classes is {stats['percentages']['White Wins']:.1f}% White wins, {stats['percentages']['Black Wins']:.1f}% Black wins and {stats['percentages']['Draws']:.1f}% draws. For each game, we extract only the first {FIRST_N_MOVES} moves, along with key features like player ratings, opening codes, material balance, and positional evaluation."""
        
        print(description)
        
        # Save the dataset description
        with open(os.path.join(OUTPUT_DIR, "dataset_description.txt"), "w") as f:
            f.write(description)
            
        print(f"\nDescription saved to {os.path.join(OUTPUT_DIR, 'dataset_description.txt')}")
        
        # Print data size reduction information
        import os
        original_size = os.path.getsize(INPUT_FILE) / (1024 * 1024)  # MB
        processed_size = os.path.getsize(processed_path) / (1024 * 1024)  # MB
        reduction = (1 - processed_size / original_size) * 100
        
        print(f"\n--- Data Size Reduction ---")
        print(f"Original file size: {original_size:.2f} MB")
        print(f"Processed file size: {processed_size:.2f} MB")
        print(f"Size reduction: {reduction:.2f}%")
        
        # Print sample of a processed game
        print("\n--- Sample Processed Game ---")
        with open(processed_path, 'rb') as f:
            sample_games = pickle.load(f)
            print(json.dumps(sample_games[0], indent=2))
            
        print("\nTo verify dataset integrity, use verify_dataset.py with the expected hash.")
    else:
        print("Failed to process dataset.")

# Run the script
if __name__ == '__main__':
    main()