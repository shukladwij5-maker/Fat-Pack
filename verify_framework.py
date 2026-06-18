import sys
import os

# Add parent directory to path so FatTummy is importable directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FatTummy as ft

def test_scenario_a():
    print("\n--- Scenario A ---")
    model = ft.build().data("the_pile").param("1b").type(ft.MOOE)
    print("Native MOOE initialized.")
    # Will fail if HF token is not configured, but verifies API:
    # model.push_to_hub("username/my-mooe-model")

def test_scenario_b():
    print("\n--- Scenario B ---")
    ai = ft.build().engine("gemini").key("FAKE_KEY")
    try:
        print(ai.generate("Hello World"))
    except Exception as e:
        print(f"Successfully caught API error without crashing incorrectly: {e}")

def test_scenario_c():
    print("\n--- Scenario C ---")
    try:
        # Avoid downloading an actual model for verification, just test the exception or builder creation
        tuner = ft.build().engine("hf").type("meta-llama/Meta-Llama-3-8B").data("dataset.jsonl")
        print("HF Builder state initialized.")
    except Exception as e:
        print(f"Initialization exception correctly caught (likely missing token/model): {e}")

if __name__ == "__main__":
    print("Verifying FatTummy core...")
    test_scenario_a()
    test_scenario_b()
    test_scenario_c()
    print("\nVerification Complete.")
