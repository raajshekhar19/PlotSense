import json

NOTEBOOK_PATH = r"d:/emerging/hybrid_search.ipynb"

def add_verification():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Verification Code Cell
    verification_source = [
        "# --- VERIFICATION ---\n",
        "print(\"Running Verification Tests for 4 Flows...\")\n",
        "\n",
        "# 1. Plot Path\n",
        "print(\"\\n1. Testing Plot Path...\")\n",
        "try:\n",
        "    res1 = app.invoke({\"query\": \"A movie about falling in love on a sinking ship\"})\n",
        "    print(f\"Result 1: {res1.get('final_answer')[:100]}...\")\n",
        "except Exception as e: print(f\"Error: {e}\")\n",
        "\n",
        "# 2. Existing Movie Path\n",
        "print(\"\\n2. Testing Existing Movie Path...\")\n",
        "try:\n",
        "    res2 = app.invoke({\"query\": \"The Matrix\"})\n",
        "    print(f\"Result 2: {res2.get('final_answer')[:100]}...\")\n",
        "except Exception as e: print(f\"Error: {e}\")\n",
        "\n",
        "# 3. Web Search Path (Non-existent movie)\n",
        "print(\"\\n3. Testing Web Search Path...\")\n",
        "try:\n",
        "    # Using a name likely not in your DB but real enough for web search\n",
        "    res3 = app.invoke({\"query\": \"Dune: Part Two\"})\n",
        "    print(f\"Result 3: {res3.get('final_answer')[:100]}...\")\n",
        "except Exception as e: print(f\"Error: {e}\")\n",
        "\n",
        "# 4. KG/Query Path\n",
        "print(\"\\n4. Testing KG Agent Path...\")\n",
        "try:\n",
        "    res4 = app.invoke({\"query\": \"Movies directed by Christopher Nolan with Sci-Fi elements\"})\n",
        "    print(f\"Result 4: {res4.get('final_answer')[:100]}...\")\n",
        "except Exception as e: print(f\"Error: {e}\")\n"
    ]

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": verification_source
    }

    nb['cells'].append(new_cell)

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Verification cell added.")

if __name__ == "__main__":
    add_verification()
