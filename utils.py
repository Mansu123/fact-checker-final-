
"""
Shared utility functions used across the application
"""
import re


def is_math_question(text: str) -> bool:
    """
    ✅ MASTER VERSION: Detect if a question is math-related
    Used by both backend.py and llm_service.py
    
    Detects:
    - Arithmetic, algebra, geometry, calculus
    - Word problems (tank/fill, train/speed, worker/work)
    - Questions with mathematical computations
    """
    
    # ✅ PRIORITY CHECK: Degree symbol with angle/geometry terms = ALWAYS math
    if '°' in text:
        geometry_terms = ['কোণ', 'angle', 'ত্রিভুজ', 'triangle', 'বৃত্ত', 'circle',
                        'সমকোণ', 'right angle', 'ব্যাসার্ধ', 'radius', 'ব্যাস', 'diameter']
        for term in geometry_terms:
            if term in text.lower():
                return True
    
    math_indicators = [
        # Bengali math terms
        'গণিত', 'হিসাব', 'সমীকরণ', 'সংখ্যা', 'যোগ', 'বিয়োগ', 'গুণ', 'ভাগ',
        'দৈর্ঘ্য', 'মিটার', 'কিমি', 'সেমি', 'বর্গ', 'ঘন', 'ক্ষেত্রফল', 'আয়তন',
        'পরিমাপ', 'সূত্র', 'শতাংশ', 'অনুপাত', 'সমানুপাতিক', 'বেগ', 'দূরত্ব',
        'সময়', 'ঘণ্টা', 'মিনিট', 'গতি', 'ত্রিভুজ', 'বৃত্ত', 'কোণ', 'সমকোণ',
        'ব্যাসার্ধ', 'ব্যাস', 'পরিধি', 'সমান্তর', 'ধারা', 'জ্যামিতি', 'অসাম্য',
        'বহুপদী', 'দ্বিঘাত', 'বীজগণিত', 'লগারিদম', 'সমাকলন', 'অবকলজ',
        'লাভ', 'ক্ষতি', 'টাকা', 'ক্রয়', 'বিক্রয়', 'মূল্য', 'পূরক', 'সম্পূরক',
        
        # English math terms
        'math', 'calculate', 'equation', 'formula', 'sum', 'difference',
        'product', 'quotient', 'length', 'width', 'height', 'area', 'volume',
        'percentage', 'ratio', 'speed', 'distance', 'time', 'geometry', 'inequality',
        'polynomial', 'quadratic', 'algebra', 'logarithm', 'calculus', 'derivative',
        'integral', 'factor', 'factoriz', 'solve for x', 'find x',
        'profit', 'loss', 'cost', 'price', 'selling', 'buying',
        'complementary', 'supplementary', 'angle', 'degree',
        
        # ✅ Word problem indicators (CRITICAL for tank/train/worker problems!)
        'tank', 'fill', 'empty', 'leak', 'tap', 'pipe', 'inlet', 'outlet', 'rate',
        'টাংকি', 'পূরণ', 'খালি', 'ট্যাপ', 'পাইপ', 'হার', 'ভরা', 'নিঃসরণ',
        'train', 'car', 'vehicle', 'travel', 'meet', 'overtake',
        'ট্রেন', 'গাড়ি', 'যান', 'ভ্রমণ', 'মিলিত',
        'worker', 'work', 'complete', 'finish', 'days', 'together',
        'কাজ', 'শ্রমিক', 'সম্পূর্ণ', 'শেষ', 'দিন', 'একসাথে',
        'age', 'years', 'old', 'father', 'son', 'mother', 'daughter',
        'বয়স', 'বছর', 'পুরানো', 'বাবা', 'ছেলে', 'মা', 'মেয়ে',
        'mixture', 'solution', 'concentration', 'mix', 'combine',
        'মিশ্রণ', 'দ্রবণ', 'ঘনত্ব', 'মিশ্র', 'মিশানো',
        'interest', 'principal', 'amount', 'compound', 'simple',
        'সুদ', 'মূলধন', 'পরিমাণ', 'চক্রবৃদ্ধি', 'সরল',
        
        # Math symbols and patterns
        '+', '×', '÷', '=', '%', '²', '³', '<', '>', '≤', '≥',
    ]
    
    text_lower = text.lower()
    
    # ✅ CHECK 1: Look for explicit math keywords
    has_math_keyword = False
    for indicator in math_indicators:
        if indicator.lower() in text_lower:
            has_math_keyword = True
            break
    
    # If no math keywords found, it's definitely NOT math
    if not has_math_keyword:
        return False
    
    # ✅ CHECK 2: If math keyword found, check for numbers
    numbers = re.findall(r'\d+', text)
    
    # Special case: Geometry/angle questions need only 1 number
    geometry_terms = ['কোণ', 'angle', 'ত্রিভুজ', 'triangle', 'বৃত্ত', 'circle',
                      'পূরক', 'সম্পূরক', 'complementary', 'supplementary']
    is_geometry = any(term in text_lower for term in geometry_terms)
    
    # ✅ RELAXED: Word problems typically have 2+ numbers
    if is_geometry and len(numbers) >= 1:
        return True
    elif len(numbers) >= 2:  # Most word problems have multiple numbers
        return True
    
    # ✅ CHECK 3: Inequality symbols with variables
    if any(symbol in text for symbol in ['<', '>', '≤', '≥', '²', '³']):
        if any(var in text for var in ['x', 'y', 'z', 'a', 'b', 'c']):
            return True
    
    return False