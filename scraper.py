"""
Universal PDF Scraper
=====================
Extracts structured data from any PDF without company-specific code.
Uses pattern recognition and table structure analysis.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pandas as pd

try:
    import camelot
    import pdfplumber
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: "
        "pip install 'camelot-py[cv]' pdfplumber pandas"
    ) from e

# ============================================================================
# Pattern Detectors
# ============================================================================

def is_person_name(text: str) -> bool:
    """Check if text looks like a person's name"""
    if not text or len(text) < 3:
        return False
    
    words = text.split()
    capitalized = [w for w in words if w and w[0].isupper()]
    
    # Should have 2-4 capitalized words
    if len(capitalized) < 2 or len(words) > 6:
        return False
    
    # Should not be ALL CAPS (likely a header)
    if text.isupper():
        return False
    
    # Should not contain numbers
    if any(c.isdigit() for c in text):
        return False
    
    return True

def extract_phones(text: str) -> List[str]:
    """Extract phone numbers from text"""
    if not text:
        return []
    
    phones = []
    # 8-digit numbers with optional formatting
    pattern = re.compile(r'(?:\+45\s*)?(\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2})')
    
    for match in pattern.finditer(str(text)):
        phone = match.group(1).replace(' ', '').replace('-', '')
        if len(phone) == 8 and phone.isdigit():
            phones.append(phone)
    
    return list(set(phones))

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text"""
    if not text:
        return []
    
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return list(set(pattern.findall(str(text))))

def is_date_like(text: str) -> bool:
    """Check if text looks like a date"""
    if not text:
        return False
    
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
    ]
    
    return any(re.search(p, text.lower()) for p in patterns)

def is_money_like(text: str) -> bool:
    """Check if text looks like money"""
    if not text:
        return False
    
    keywords = ['mio', 'mia', 'kr', 'dkk', 'euro', '€', '$', 'million', 'billion']
    return any(k in text.lower() for k in keywords)

# ============================================================================
# Column Analysis
# ============================================================================

def detect_column_type(series: pd.Series) -> str:
    """
    Detect what type of data a column contains.
    Returns: 'name', 'phone', 'email', 'date', 'money', 'role', 'id', 'text', 'unknown'
    """
    
    samples = series.dropna().astype(str).head(20)
    if len(samples) == 0:
        return 'unknown'
    
    # Count patterns
    name_count = sum(is_person_name(s) for s in samples)
    phone_count = sum(len(extract_phones(s)) > 0 for s in samples)
    email_count = sum(len(extract_emails(s)) > 0 for s in samples)
    date_count = sum(is_date_like(s) for s in samples)
    money_count = sum(is_money_like(s) for s in samples)
    
    total = len(samples)
    
    # Determine type by majority
    if email_count / total > 0.3:
        return 'email'
    elif phone_count / total > 0.3:
        return 'phone'
    elif name_count / total > 0.5:
        return 'name'
    elif date_count / total > 0.5:
        return 'date'
    elif money_count / total > 0.3:
        return 'money'
    elif all(s.isdigit() for s in samples if s) and len(samples[0]) < 5:
        return 'id'
    elif all(len(s) < 200 for s in samples):
        return 'text'
    
    return 'unknown'

# ============================================================================
# Table Classification
# ============================================================================

def classify_table(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Classify table type and confidence.
    Returns: (type, confidence) where type is 'contact', 'project', 'tender', or 'unknown'
    """
    
    if df.empty or len(df) < 2:
        return ('unknown', 0.0)
    
    # Detect column types
    column_types = {col: detect_column_type(df[col]) for col in df.columns}
    type_counts = Counter(column_types.values())
    
    # Score different table types
    contact_score = 0.0
    project_score = 0.0
    tender_score = 0.0
    
    # Contact table: has name + phone/email
    if type_counts.get('name', 0) >= 1:
        contact_score += 3.0
    if type_counts.get('phone', 0) >= 1:
        contact_score += 2.0
    if type_counts.get('email', 0) >= 1:
        contact_score += 2.0
    
    # Project table: has text/name + money + dates
    if type_counts.get('text', 0) >= 1 or type_counts.get('name', 0) >= 1:
        project_score += 1.0
    if type_counts.get('money', 0) >= 1:
        project_score += 3.0
    if type_counts.get('date', 0) >= 2:
        project_score += 2.0
    
    # Tender table: has dates + text
    if type_counts.get('date', 0) >= 2:
        tender_score += 2.0
    if 'bid' in str(df.columns).lower() or 'tender' in str(df.columns).lower():
        tender_score += 3.0
    
    max_score = max(contact_score, project_score, tender_score)
    
    if max_score == 0:
        return ('unknown', 0.0)
    
    confidence = min(max_score / 8.0, 1.0)
    
    if contact_score == max_score:
        return ('contact', confidence)
    elif project_score == max_score:
        return ('project', confidence)
    else:
        return ('tender', confidence)

# ============================================================================
# Data Extraction
# ============================================================================

def extract_row_data(row: pd.Series, column_types: Dict[str, str]) -> Optional[Dict]:
    """Extract data from a table row based on column types"""
    
    data = {}
    confidence = 1.0
    
    for col, col_type in column_types.items():
        value = str(row[col]).strip() if pd.notna(row[col]) else ""
        
        if not value or value == 'nan':
            continue
        
        if col_type == 'name':
            if is_person_name(value):
                data['name'] = value
            else:
                confidence *= 0.8
        
        elif col_type == 'phone':
            phones = extract_phones(value)
            if phones:
                data['phone'] = phones[0]
        
        elif col_type == 'email':
            emails = extract_emails(value)
            if emails:
                data['email'] = emails[0]
        
        elif col_type == 'date':
            if 'dates' not in data:
                data['dates'] = []
            data['dates'].append(value)
        
        elif col_type == 'money':
            data['budget'] = value
        
        elif col_type == 'text':
            if 'name' not in data and len(value) > 5:
                data['description'] = value
        
        elif col_type == 'id':
            data['id'] = value
    
    # Must have meaningful data
    if not data or len(data) < 1:
        return None
    
    # Filter very low confidence
    if confidence < 0.3:
        return None
    
    data['_confidence'] = confidence
    return data

def extract_table_data(df: pd.DataFrame, table_type: str) -> List[Dict]:
    """Extract all valid rows from a table"""
    
    # Detect column types
    column_types = {col: detect_column_type(df[col]) for col in df.columns}
    
    # Extract rows
    results = []
    for _, row in df.iterrows():
        data = extract_row_data(row, column_types)
        
        if data and data.get('_confidence', 0) >= 0.5:
            # Remove internal confidence field
            confidence = data.pop('_confidence', 0)
            
            # Validate based on table type
            if table_type == 'contact':
                # Must have name and at least phone or email
                if 'name' in data and ('phone' in data or 'email' in data):
                    results.append(data)
            elif table_type == 'project':
                # Must have description or name
                if 'description' in data or 'name' in data:
                    results.append(data)
            elif table_type == 'tender':
                results.append(data)
    
    return results

# ============================================================================
# Table Extraction
# ============================================================================

def extract_all_tables(pdf_path: str) -> List[Tuple[pd.DataFrame, int]]:
    """
    Extract all tables from PDF using multiple methods.
    Returns: List of (dataframe, page_number)
    """
    
    all_tables = []
    
    # Method 1: Camelot lattice (tables with borders)
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 1:
                all_tables.append((table.df, table.page))
    except:
        pass
    
    # Method 2: Camelot stream (tables without borders)
    try:
        tables = camelot.read_pdf(
            pdf_path, 
            pages='all', 
            flavor='stream',
            edge_tol=50,
            row_tol=10
        )
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 1:
                # Check for duplicates
                is_dup = False
                for existing_df, _ in all_tables:
                    if existing_df.shape == table.df.shape:
                        is_dup = True
                        break
                
                if not is_dup:
                    all_tables.append((table.df, table.page))
    except:
        pass
    
    # Method 3: pdfplumber (fallback)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_data in tables:
                    if table_data and len(table_data) > 1:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        if not df.empty:
                            all_tables.append((df, page_num))
    except:
        pass
    
    return all_tables

# ============================================================================
# Company Info
# ============================================================================

def extract_company_info(pdf_path: str) -> Dict[str, str]:
    """Extract company information from first page"""
    
    info = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return info
            
            text = pdf.pages[0].extract_text() or ""
            lines = text.split('\n')[:30]
            
            for line in lines:
                line = line.strip()
                
                # CVR (Danish company registration)
                if 'cvr' in line.lower():
                    match = re.search(r'\b(\d{8})\b', line)
                    if match:
                        info['cvr'] = match.group(1)
                
                # Email
                emails = extract_emails(line)
                if emails and 'email' not in info:
                    info['email'] = emails[0]
                
                # Website
                if 'http' in line.lower():
                    match = re.search(r'https?://[^\s]+', line)
                    if match:
                        info['website'] = match.group(0)
                
                # Phone
                if 'phone' not in info and 'telefon' in line.lower():
                    phones = extract_phones(line)
                    if phones:
                        info['phone'] = phones[0]
                
                # Company name
                if 'name' not in info:
                    if any(suffix in line for suffix in [' A/S', ' ApS', ' A.S', ' IVS']):
                        if len(line) < 60 and not line.isupper():
                            info['name'] = line
    
    except:
        pass
    
    return info

# ============================================================================
# Main Parser
# ============================================================================

def parse_pdf(pdf_path: str) -> Dict:
    """
    Parse PDF and extract all structured data.
    
    Returns dict with:
    - company_info: Company details
    - contacts: List of contacts
    - projects: List of projects  
    - tenders: List of tenders
    - quality: Quality metrics
    - summary: Extraction summary
    """
    
    # Extract company info
    company_info = extract_company_info(pdf_path)
    
    # Extract all tables
    raw_tables = extract_all_tables(pdf_path)
    
    # Process tables
    contacts = []
    projects = []
    tenders = []
    quality_scores = []
    
    for df, page in raw_tables:
        # Classify table
        table_type, confidence = classify_table(df)
        
        if table_type == 'unknown' or confidence < 0.3:
            continue
        
        quality_scores.append(confidence)
        
        # Extract data
        rows = extract_table_data(df, table_type)
        
        if table_type == 'contact':
            contacts.extend(rows)
        elif table_type == 'project':
            projects.extend(rows)
        elif table_type == 'tender':
            tenders.extend(rows)
    
    # Calculate quality metrics
    avg_confidence = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return {
        'company_info': company_info,
        'contacts': contacts,
        'projects': projects,
        'tenders': tenders,
        'quality': {
            'avg_confidence': round(avg_confidence, 2),
            'tables_processed': len(quality_scores),
            'extraction_method': 'multi-method (camelot + pdfplumber)'
        },
        'summary': {
            'contacts': len(contacts),
            'projects': len(projects),
            'tenders': len(tenders)
        }
    }
