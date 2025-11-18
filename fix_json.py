
#!/usr/bin/env python3
"""
Fix malformed JSON - specifically for large dataset files
"""
import json
import sys

def fix_json_file(input_path, output_path):
    print("="*70)
    print("JSON FILE FIXER")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Read the entire file
    print("\nReading file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File size: {len(content)} bytes")
    
    # Try to find and fix the issue
    print("\nLooking for common JSON errors...")
    
    # Fix 1: Remove trailing commas before ] or }
    import re
    original_content = content
    content = re.sub(r',(\s*[\]}])', r'\1', content)
    
    if content != original_content:
        print("✓ Fixed trailing commas")
    
    # Try to parse
    print("\nAttempting to parse JSON...")
    try:
        data = json.loads(content)
        print(f"✓ SUCCESS! Loaded {len(data)} items")
        
        # Save fixed version
        print(f"\nSaving to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✓ Saved successfully!")
        print("\n" + "="*70)
        print("NEXT STEP:")
        print("="*70)
        print(f"Update your .env file:")
        print(f"DATASET_PATH={output_path}")
        print("\nThen run: python data_preprocessing.py")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Still has error: {e}")
        print(f"\nError at line {e.lineno}, column {e.colno}")
        print(f"Character position: {e.pos}")
        
        # Show context around error
        lines = content.split('\n')
        error_line = e.lineno - 1
        
        print("\nContext around error:")
        print("-"*70)
        for i in range(max(0, error_line - 3), min(len(lines), error_line + 4)):
            marker = ">>> " if i == error_line else "    "
            print(f"{marker}Line {i+1}: {lines[i][:100]}")
        print("-"*70)
        
        # Try to manually fix the specific issue
        print("\nAttempting manual fix...")
        
        # Check if it's a trailing comma issue
        if error_line < len(lines):
            problem_line = lines[error_line]
            
            # If the error is expecting a property name, might be extra comma
            if "Expecting property name" in str(e):
                print("Issue: Extra comma or missing property")
                
                # Find the problematic part and try to fix
                # Remove the line with the issue
                lines[error_line] = ""
                content = '\n'.join(lines)
                
                # Try again
                try:
                    data = json.loads(content)
                    print(f"✓ FIXED! Loaded {len(data)} items")
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    print("✓ Saved successfully!")
                    return True
                except:
                    pass
        
        print("\n✗ Automatic fix failed")
        print("\nManual fix needed:")
        print(f"1. Open file: {input_path}")
        print(f"2. Go to line: {e.lineno}")
        print(f"3. Look for:")
        print(f"   - Extra commas")
        print(f"   - Missing quotes")
        print(f"   - Unclosed brackets")
        
        return False

if __name__ == "__main__":
    input_file = "/Users/mansuba/Desktop/fq fact check/fact check and question generation project/data/favourite_question_40k.json"
    output_file = "/Users/mansuba/Desktop/fq fact check/fact check and question generation project/data/favourite_question_40k_fixed.json"
    
    success = fix_json_file(input_file, output_file)
    
    sys.exit(0 if success else 1)