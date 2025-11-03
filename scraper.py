"""
Enhanced Professional PDF Scraper
===================================
Optimized for Danish construction company PDFs with:
- Multi-line cell handling
- Danish pattern recognition  
- Better column mapping
- Proper contact/project/tender separation
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
import logging
import numpy as np

try:
    import camelot
    import pdfplumber
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: "
        "pip install 'camelot-py[cv]' pdfplumber pandas"
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Utilities
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_multiline(text: str) -> str:
    """Clean multi-line cell text but preserve meaningful line breaks"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove excessive whitespace but keep single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

# ============================================================================
# Pattern Detection
# ============================================================================

def is_valid_person_name(text: str) -> bool:
    """
    Validate person names (2-4 capitalized words, no numbers, no roles)
    """
    if not text or len(text) < 3 or len(text) > 70:
        return False
    
    text = clean_text(text)
    
    # Filter out common non-names
    blacklist = [
        'projekt', 'kontakt', 'entr', 'entrepren', 'rådgiver', 'ingeniør',
        'chef', 'direktør', 'a/s', 'aps', 'firma', 'rolle', 'telefon',
        'navn', 'cvr', 'total', 'hoved', 'bygge', 'element', 'beton',
        'tømrer', 'snedker', 'murer', 'maler', 'elektriker', 'vvs',
        'tagdækning', 'facade', 'gulv', 'vindue', 'dør', 'stål', 'smede'
    ]
    
    text_lower = text.lower()
    if any(word in text_lower for word in blacklist):
        return False
    
    # Must not be all caps (headers)
    if text.isupper() and len(text) > 8:
        return False
    
    # Split into words
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False
    
    # Each word should start with capital
    capitalized = [w for w in words if w and len(w) > 0 and w[0].isupper()]
    if len(capitalized) < 2:
        return False
    
    # No numbers
    if any(char.isdigit() for char in text):
        return False
    
    # Each word mostly letters
    for word in words:
        if len(word) > 1:
            letters = sum(1 for c in word if c.isalpha())
            if letters < len(word) * 0.7:
                return False
    
    return True

def extract_phones(text: str) -> List[str]:
    """Extract Danish phone numbers (8 digits)"""
    if not text or pd.isna(text):
        return []
    
    text = str(text)
    phones = []
    
    # Pattern for 8-digit Danish numbers
    patterns = [
        r'(?:\+45\s*)?(\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2})',
        r'\b(\d{8})\b'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            phone = re.sub(r'[\s\-]', '', match.group(1))
            if len(phone) == 8 and phone.isdigit():
                # Avoid dates that look like phone numbers
                if not re.search(r'(19|20)\d{2}', phone):
                    phones.append(phone)
    
    return list(dict.fromkeys(phones))  # Remove duplicates, preserve order

def extract_emails(text: str) -> List[str]:
    """Extract email addresses"""
    if not text or pd.isna(text):
        return []
    
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return list(set(pattern.findall(str(text))))

def extract_roles(text: str) -> List[str]:
    """Extract job roles/positions from text"""
    if not text or pd.isna(text):
        return []
    
    text = clean_multiline(text)
    roles = []
    
    # Common role patterns
    role_patterns = [
        r'Projektleder[^.]*',
        r'Kontaktperson[^.]*',
        r'Byggeleder[^.]*',
        r'Sagsansvarlig[^.]*',
        r'Projektchef[^.]*',
        r'Ingeniør[^.]*',
        r'Rådg\.ing\.[^.]*',
        r'Totalentreprenør',
        r'Hovedentreprenør',
        r'El-entreprenør',
        r'[A-ZÆØÅ][a-zæøå]+ Entr\.',
    ]
    
    for pattern in role_patterns:
        matches = re.findall(pattern, text)
        roles.extend(matches)
    
    # Clean and deduplicate
    cleaned = []
    for role in roles:
        role = clean_text(role)
        if role and role not in cleaned:
            cleaned.append(role)
    
    return cleaned[:5]  # Max 5 roles

# ============================================================================
# Table Classification
# ============================================================================

def detect_table_type(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Detect if table contains contacts, projects, or tenders
    Returns: (type, confidence)
    """
    
    if df.empty or len(df) < 2:
        return ('unknown', 0.0)
    
    # Get all text from table
    all_text = ' '.join(
        str(cell).lower() 
        for row in df.values 
        for cell in row 
        if pd.notna(cell)
    )
    
    # Contact table indicators
    contact_score = 0.0
    if 'navn' in all_text:
        contact_score += 2.0
    if any(word in all_text for word in ['telefon', 'phone', 'mobil']):
        contact_score += 3.0
    if 'email' in all_text or 'e-mail' in all_text:
        contact_score += 2.0
    if 'rolle' in all_text or 'kontaktperson' in all_text:
        contact_score += 2.0
    if 'firma' in all_text:
        contact_score += 1.0
    
    # Check if we have actual person names
    name_count = 0
    for row in df.values[:20]:  # Sample first 20 rows
        for cell in row:
            if pd.notna(cell) and is_valid_person_name(str(cell)):
                name_count += 1
    
    if name_count >= 5:
        contact_score += 3.0
    
    # Project table indicators
    project_score = 0.0
    if 'projekt' in all_text:
        project_score += 3.0
    if any(word in all_text for word in ['budget', 'mio', 'kr']):
        project_score += 3.0
    if any(word in all_text for word in ['byggestart', 'dato', 'date']):
        project_score += 2.0
    if 'region' in all_text or 'hovedstaden' in all_text:
        project_score += 2.0
    if any(word in all_text for word in ['stadie', 'udførelse', 'udbud']):
        project_score += 2.0
    
    # Tender table indicators
    tender_score = 0.0
    if 'udbud' in all_text:
        tender_score += 5.0
    if 'licitation' in all_text:
        tender_score += 3.0
    if all_text.count('arkiv') >= 3:  # Multiple archived tenders
        tender_score += 2.0
    
    # Determine type
    scores = {'contact': contact_score, 'project': project_score, 'tender': tender_score}
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]
    
    if max_score < 3.0:
        return ('unknown', 0.0)
    
    confidence = min(max_score / 10.0, 1.0)
    return (max_type, confidence)

# ============================================================================
# Contact Extraction
# ============================================================================

def find_column_indices(df: pd.DataFrame, keywords: List[str]) -> List[int]:
    """Find column indices that contain any of the keywords"""
    indices = []
    
    # Check first few rows for headers
    for col_idx in range(len(df.columns)):
        col_text = ' '.join(
            str(df.iloc[i, col_idx]).lower() 
            for i in range(min(5, len(df))) 
            if pd.notna(df.iloc[i, col_idx])
        )
        
        if any(kw in col_text for kw in keywords):
            indices.append(col_idx)
    
    return indices

def merge_multirow_entries(df: pd.DataFrame, boundary_cols: List[int]) -> pd.DataFrame:
    """
    UNIVERSAL: Merge multi-row entries (contacts, projects, etc.) into single rows.
    
    Many PDFs split entries across multiple rows. This function detects and merges them.
    
    Args:
        df: DataFrame to process
        boundary_cols: Columns to check for entry boundaries (ID, name, etc.)
    
    Strategy:
    1. If ID column exists: Use numbers as entry boundaries (most reliable)
    2. Otherwise: Use presence of text in boundary columns
    """
    if df.empty or not boundary_cols:
        return df
    
    # Detect ID column (universal pattern)
    id_col = detect_id_column(df)
    primary_col = boundary_cols[0]
    
    merged_rows = []
    current_entry = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Check if this row starts a new entry
        is_new_entry = False
        
        if id_col is not None:
            # UNIVERSAL: Use ID column as delimiter (most reliable)
            cell_val = str(row.iloc[id_col]).strip()
            if cell_val.isdigit():
                is_new_entry = True
        else:
            # Fallback: Use primary column presence
            cell = str(row.iloc[primary_col]) if primary_col < len(row) else ""
            is_new_entry = cell.strip() and cell not in ['', 'nan', 'None']
        
        if is_new_entry:
            # Save previous entry
            if current_entry is not None:
                merged_rows.append(current_entry)
            # Start new entry
            current_entry = row.copy()
        else:
            # Continuation of previous entry - merge cells
            if current_entry is not None:
                for col_idx in range(len(row)):
                    cell_value = str(row.iloc[col_idx]).strip()
                    if cell_value and cell_value not in ['', 'nan', 'None']:
                        # Append to current entry's cell with newline separator
                        existing = str(current_entry.iloc[col_idx]).strip()
                        if not existing or existing in ['', 'nan', 'None']:
                            current_entry.iloc[col_idx] = cell_value
                        else:
                            current_entry.iloc[col_idx] = existing + '\n' + cell_value
    
    # Don't forget last entry
    if current_entry is not None:
        merged_rows.append(current_entry)
    
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        id_detected = id_col is not None
        logger.info(f"Merged {len(df)} rows into {len(merged_df)} entries (ID column: {id_detected})")
        return merged_df
    
    return df

def merge_multirow_contacts(df: pd.DataFrame, name_cols: List[int]) -> pd.DataFrame:
    """
    UNIVERSAL: Merge multi-row contact entries.
    Wrapper around merge_multirow_entries for backward compatibility.
    """
    return merge_multirow_entries(df, name_cols)

def detect_id_column(df: pd.DataFrame) -> Optional[int]:
    """
    Detect if table has a number/ID column (common pattern in PDFs).
    
    Returns column index if found, None otherwise.
    """
    for col_idx in range(min(3, len(df.columns))):  # Check first 3 columns
        # Check first 10 rows for numeric sequence pattern
        numbers = []
        for i in range(min(10, len(df))):
            cell = str(df.iloc[i, col_idx]).strip()
            if cell.isdigit() and len(cell) <= 3:  # IDs typically 1-3 digits
                numbers.append(int(cell))
        
        # If we found 3+ sequential-ish numbers, likely an ID column
        if len(numbers) >= 3:
            # Check if mostly sequential (allows some gaps)
            sorted_nums = sorted(numbers)
            if sorted_nums[-1] - sorted_nums[0] <= len(numbers) * 2:
                logger.info(f"Detected ID column at index {col_idx}")
                return col_idx
    
    return None

def extract_contacts_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract contacts with names, phones, emails, and roles"""
    
    contacts = []
    
    logger.info(f"Extracting contacts from table with shape {df.shape}")
    
    # Find relevant columns
    name_cols = find_column_indices(df, ['navn', 'name'])
    phone_cols = find_column_indices(df, ['telefon', 'phone', 'mobil'])
    email_cols = find_column_indices(df, ['email', 'e-mail', 'mail'])
    role_cols = find_column_indices(df, ['rolle', 'position', 'titel'])
    
    # If no explicit columns found, try to infer
    if not name_cols:
        # Look for column with most person names
        name_counts = []
        for col_idx in range(len(df.columns)):
            count = sum(
                1 for i in range(len(df)) 
                if pd.notna(df.iloc[i, col_idx]) and 
                is_valid_person_name(str(df.iloc[i, col_idx]))
            )
            name_counts.append((col_idx, count))
        
        if name_counts:
            best_col, best_count = max(name_counts, key=lambda x: x[1])
            if best_count >= 3:
                name_cols = [best_col]
    
    if not name_cols:
        logger.warning("No name column found in contact table")
        return []
    
    logger.info(f"Column indices - names: {name_cols}, phones: {phone_cols}, emails: {email_cols}, roles: {role_cols}")
    
    # UNIVERSAL FIX: Merge multi-row contacts before processing
    df = merge_multirow_contacts(df, name_cols)
    
    # Detect ID column for extraction
    id_col = detect_id_column(df)
    
    # Skip header rows
    start_row = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if any(kw in row_text for kw in ['navn', 'name', 'firma', 'telefon', 'rolle']):
            start_row = i + 1
    
    # Extract contacts
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        
        contact = {}
        
        # Extract ID/number if column exists
        if id_col is not None and id_col < len(row):
            contact_id = str(row.iloc[id_col]).strip()
            if contact_id.isdigit():
                contact['id'] = contact_id
        
        # Extract name
        for name_col in name_cols:
            if name_col < len(row):
                name = clean_text(str(row.iloc[name_col]))
                if name and is_valid_person_name(name):
                    contact['name'] = name
                    break
        
        if 'name' not in contact:
            continue
        
        # Extract phone - check phone columns and all columns
        for col_idx in phone_cols + list(range(len(row))):
            if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                phones = extract_phones(str(row.iloc[col_idx]))
                if phones:
                    contact['phone'] = phones[0]
                    break
        
        # Extract email - check email columns and all columns
        for col_idx in email_cols + list(range(len(row))):
            if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                emails = extract_emails(str(row.iloc[col_idx]))
                if emails:
                    contact['email'] = emails[0]
                    break
        
        # Extract roles - check role columns and all columns
        all_roles = []
        for col_idx in role_cols + list(range(len(row))):
            if col_idx < len(row) and col_idx not in name_cols:
                cell_text = str(row.iloc[col_idx])
                if pd.notna(row.iloc[col_idx]):
                    roles = extract_roles(cell_text)
                    all_roles.extend(roles)
        
        if all_roles:
            contact['roles'] = list(dict.fromkeys(all_roles))[:3]  # Max 3 roles
        
        # UNIVERSAL: Include contact if has name + (phone OR email OR role)
        # Many PDFs have contacts with roles but no direct phone/email
        if 'phone' in contact or 'email' in contact or 'roles' in contact:
            contacts.append(contact)
    
    # Deduplicate
    seen = set()
    unique = []
    for contact in contacts:
        key = (contact.get('name', ''), contact.get('phone', ''), contact.get('email', ''))
        if key not in seen:
            seen.add(key)
            unique.append(contact)
    
    logger.info(f"Extracted {len(unique)} unique contacts")
    return unique

# ============================================================================
# Project Extraction
# ============================================================================

def extract_budget(text: str) -> Optional[str]:
    """Extract budget value from text"""
    if not text or pd.isna(text):
        return None
    
    text = clean_multiline(text)
    
    # Look for patterns like "155 mio", "3,4 mia", "900 mio. kr"
    patterns = [
        r'(\d+(?:[,.]\d+)?\s*(?:mia|mio)\.?\s*(?:kr)?)',
        r'(\d+(?:[,.]\d+)?\s*billion)',
        r'(\d+(?:[,.]\d+)?\s*million)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    
    return None

def extract_date(text: str) -> Optional[str]:
    """Extract date from text"""
    if not text or pd.isna(text):
        return None
    
    text = clean_multiline(text)
    
    # Danish month patterns
    month_pattern = r'(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)[a-z]*\.?\s+\d{4}'
    match = re.search(month_pattern, text, re.IGNORECASE)
    if match:
        return clean_text(match.group(0))
    
    # Numeric date pattern
    date_pattern = r'\d{1,2}[-./]\d{1,2}[-./]\d{4}'
    match = re.search(date_pattern, text)
    if match:
        return clean_text(match.group(0))
    
    # Year only
    year_pattern = r'\b(20\d{2})\b'
    match = re.search(year_pattern, text)
    if match:
        return match.group(1)
    
    return None

def extract_region(text: str) -> Optional[str]:
    """Extract Danish region from text"""
    if not text or pd.isna(text):
        return None
    
    regions = ['Hovedstaden', 'Sjælland', 'Syddanmark', 'Midtjylland', 'Nordjylland']
    text_clean = clean_multiline(text)
    
    for region in regions:
        if region in text_clean:
            return region
    
    return None

def extract_stage(text: str) -> Optional[str]:
    """Extract project stage/status"""
    if not text or pd.isna(text):
        return None
    
    stages = [
        'Udførelsesproces',
        'Udbudsproces', 
        'Projekteringsproces',
        'Planlægningsproces',
        'Afsluttet',
        'Skitseprojekt'
    ]
    
    text_clean = clean_multiline(text)
    
    for stage in stages:
        if stage.lower() in text_clean.lower():
            return stage
    
    return None

def extract_projects_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract projects with all details"""
    
    projects = []
    
    logger.info(f"Extracting projects from table with shape {df.shape}")
    
    # Skip header rows
    start_row = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if any(kw in row_text for kw in ['projekt', 'budget', 'region', 'rolle']):
            start_row = i + 1
    
    # Extract data portion (skip headers)
    df_data = df.iloc[start_row:].copy() if start_row < len(df) else df.copy()
    df_data = df_data.reset_index(drop=True)
    
    # UNIVERSAL FIX: Merge multi-row projects before processing
    # Use first column as boundary (usually project name or ID)
    df_data = merge_multirow_entries(df_data, [0])
    
    # Detect ID column for extraction
    id_col = detect_id_column(df_data)
    
    # Process each merged row
    # Process each merged row
    for idx in range(len(df_data)):
        row = df_data.iloc[idx]
        
        project = {}
        
        # Extract ID/number if column exists
        if id_col is not None and id_col < len(row):
            project_id = str(row.iloc[id_col]).strip()
            # Clean up any newlines from merging
            project_id = project_id.split('\n')[0].strip()
            if project_id.isdigit():
                project['id'] = project_id
        
        # Collect all non-empty cells
        cells = [clean_multiline(str(cell)) for cell in row if pd.notna(cell) and str(cell).strip()]
        
        if not cells:
            continue
        
        # Extract project name - usually the first or longest text field
        project_name = None
        for cell in cells:
            # Filter out pure metadata
            if len(cell) > 15 and not re.match(r'^\d+\s+(mio|mia)', cell.lower()):
                # Check if it's not just a role or region
                if not any(word in cell.lower() for word in ['hovedstaden', 'sjælland', 'entr.', 'totalentreprenør']):
                    if not re.match(r'^\d{1,2}\s+\w+\.?\s+\d{4}', cell):  # Not a date
                        project_name = cell
                        break
        
        if not project_name:
            # Try first cell that's not empty
            for cell in cells:
                if len(cell) > 10:
                    project_name = cell
                    break
        
        if not project_name:
            continue
        
        # Clean project name - replace newlines with spaces (from multi-row merging)
        project_name = re.sub(r'\s*\n\s*', ' ', project_name)
        project_name = re.sub(r'\s+', ' ', project_name).strip()
        
        project['name'] = project_name
        
        # Extract other fields from all cells
        all_text = ' '.join(cells)
        
        budget = extract_budget(all_text)
        if budget:
            project['budget'] = budget
        
        date = extract_date(all_text)
        if date:
            project['start_date'] = date
        
        region = extract_region(all_text)
        if region:
            project['region'] = region
        
        stage = extract_stage(all_text)
        if stage:
            project['stage'] = stage
        
        # Extract roles (like Hovedentreprenør, Totalentreprenør)
        roles = extract_roles(all_text)
        if roles:
            project['roles'] = roles[:2]  # Max 2 roles
        
        # Only add if we have meaningful data
        if len(project) >= 2:  # At least name + one other field
            projects.append(project)
    
    # Deduplicate based on name
    seen = set()
    unique = []
    for project in projects:
        name = project.get('name', '')
        if name and name not in seen:
            seen.add(name)
            unique.append(project)
    
    logger.info(f"Extracted {len(unique)} unique projects")
    return unique

# ============================================================================
# Tender Extraction
# ============================================================================

def extract_tenders_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract tender/bid information"""
    
    tenders = []
    
    logger.info(f"Extracting tenders from table with shape {df.shape}")
    
    # Skip header rows
    start_row = 0
    for i in range(min(5, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if 'udbud' in row_text or 'licitation' in row_text:
            start_row = i + 1
    
    # Process rows
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        
        tender = {}
        
        cells = [clean_text(str(cell)) for cell in row if pd.notna(cell) and str(cell).strip()]
        
        if not cells:
            continue
        
        # Extract tender name
        for cell in cells:
            if len(cell) > 10 and 'arkiv' not in cell.lower():
                tender['name'] = cell
                break
        
        if 'name' not in tender and cells:
            tender['name'] = cells[0]
        
        # Extract trade/type (VVS Entr., El-entreprenør, etc.)
        all_text = ' '.join(cells)
        roles = extract_roles(all_text)
        if roles:
            tender['trade'] = roles[0]
        
        # Extract dates
        date = extract_date(all_text)
        if date:
            tender['date'] = date
        
        # Mark as archived if indicated
        if 'arkiv' in all_text.lower():
            tender['status'] = 'Archived'
        
        if 'name' in tender:
            tenders.append(tender)
    
    logger.info(f"Extracted {len(tenders)} tenders")
    return tenders

# ============================================================================
# Main Parser
# ============================================================================

def parse_pdf(pdf_path: str) -> Dict:
    """
    Enhanced PDF parsing with proper table classification
    """
    
    logger.info(f"Parsing PDF: {pdf_path}")
    
    # Extract company info
    company_info = extract_company_info(pdf_path)
    
    # Extract all tables with Camelot
    all_tables = []
    
    # Method 1: Lattice (bordered tables)
    try:
        logger.info("Extracting tables with Camelot lattice...")
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice', strip_text='\n')
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 2:
                all_tables.append((
                    table.df, 
                    table.page, 
                    'lattice',
                    table.parsing_report.get('accuracy', 0)
                ))
        logger.info(f"Found {len(tables)} lattice tables")
    except Exception as e:
        logger.warning(f"Camelot lattice failed: {e}")
    
    # Method 2: Stream (borderless tables) - try multiple configurations
    try:
        logger.info("Extracting tables with Camelot stream...")
        
        # Try multiple parameter combinations to handle various table structures
        stream_configs = [
            {},                                                     # Default - often best
            {'edge_tol': 50, 'row_tol': 10, 'column_tol': 5},   # Tighter - better columns
            {'edge_tol': 100, 'row_tol': 15, 'column_tol': 10}, # Moderate
            {'edge_tol': 200, 'row_tol': 20, 'column_tol': 15}, # Looser
        ]
        
        best_tables = []
        best_score = 0
        
        for config in stream_configs:
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='stream',
                    **config
                )
                
                # Score based on: number of tables + average columns + accuracy
                score = len(tables)
                if tables:
                    avg_cols = sum(t.df.shape[1] for t in tables) / len(tables)
                    avg_acc = sum(t.parsing_report.get('accuracy', 0) for t in tables) / len(tables)
                    score = score * avg_cols * (avg_acc / 100)
                
                if score > best_score:
                    best_score = score
                    best_tables = tables
                    logger.info(f"Better extraction with {config}: {len(tables)} tables, score={score:.1f}")
                    
            except Exception as e:
                logger.debug(f"Stream config {config} failed: {e}")
                continue
        
        tables = best_tables
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 2:
                # Check for duplicates
                is_dup = False
                for existing_df, _, _, _ in all_tables:
                    if existing_df.shape == table.df.shape:
                        # Check if content is similar
                        if np.array_equal(existing_df.values, table.df.values):
                            is_dup = True
                            break
                
                if not is_dup:
                    all_tables.append((
                        table.df,
                        table.page,
                        'stream',
                        table.parsing_report.get('accuracy', 0)
                    ))
        logger.info(f"Found {len(tables)} stream tables")
    except Exception as e:
        logger.warning(f"Camelot stream failed: {e}")
    
    logger.info(f"Total tables to process: {len(all_tables)}")
    
    # Process and classify tables
    contacts = []
    projects = []
    tenders = []
    quality_scores = []
    
    for df, page, method, accuracy in all_tables:
        # Detect table type
        table_type, confidence = detect_table_type(df)
        
        logger.info(
            f"Page {page}, {method}: {table_type} "
            f"(confidence: {confidence:.2f}, accuracy: {accuracy:.0f}%, shape: {df.shape})"
        )
        
        if table_type == 'unknown' or confidence < 0.3:
            continue
        
        quality_scores.append(confidence)
        
        # Extract data based on type
        if table_type == 'contact':
            new_contacts = extract_contacts_from_table(df)
            contacts.extend(new_contacts)
            
        elif table_type == 'project':
            new_projects = extract_projects_from_table(df)
            projects.extend(new_projects)
            
        elif table_type == 'tender':
            new_tenders = extract_tenders_from_table(df)
            tenders.extend(new_tenders)
    
    # Final deduplication
    contacts = deduplicate_contacts(contacts)
    projects = deduplicate_projects(projects)
    
    # Calculate quality metrics
    avg_confidence = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    logger.info(
        f"Final results: {len(contacts)} contacts, "
        f"{len(projects)} projects, {len(tenders)} tenders"
    )
    
    return {
        'company_info': company_info,
        'contacts': contacts,
        'projects': projects,
        'tenders': tenders,
        'quality': {
            'avg_confidence': round(avg_confidence, 2),
            'tables_processed': len(quality_scores),
            'extraction_method': 'enhanced-camelot'
        },
        'summary': {
            'contacts': len(contacts),
            'projects': len(projects),
            'tenders': len(tenders)
        }
    }

# ============================================================================
# Deduplication
# ============================================================================

def deduplicate_contacts(contacts: List[Dict]) -> List[Dict]:
    """Remove duplicate contacts"""
    seen = set()
    unique = []
    
    for contact in contacts:
        # Create key from name and phone/email
        key_parts = [contact.get('name', '')]
        if contact.get('phone'):
            key_parts.append(contact['phone'])
        if contact.get('email'):
            key_parts.append(contact['email'])
        
        key = tuple(key_parts)
        
        if key not in seen:
            seen.add(key)
            unique.append(contact)
    
    return unique

def deduplicate_projects(projects: List[Dict]) -> List[Dict]:
    """Remove duplicate projects"""
    seen = set()
    unique = []
    
    for project in projects:
        name = project.get('name', '').lower()
        
        # Create a normalized key
        key = re.sub(r'\s+', ' ', name).strip()
        
        if key and key not in seen:
            seen.add(key)
            unique.append(project)
    
    return unique

# ============================================================================
# Company Info Extraction
# ============================================================================

def extract_company_info(pdf_path: str) -> Dict[str, str]:
    """Extract company information from first page"""
    
    info = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return info
            
            # Extract text from first page
            text = pdf.pages[0].extract_text() or ""
            lines = text.split('\n')[:40]  # First 40 lines
            
            for line in lines:
                line = line.strip()
                
                # CVR number
                if 'cvr' in line.lower():
                    match = re.search(r'\b(\d{8})\b', line)
                    if match:
                        info['cvr'] = match.group(1)
                
                # Email
                if 'email' not in info:
                    emails = extract_emails(line)
                    if emails:
                        info['email'] = emails[0]
                
                # Website
                if 'http' in line.lower():
                    match = re.search(r'(https?://[^\s]+)', line)
                    if match:
                        info['website'] = match.group(1)
                
                # Phone
                if 'phone' not in info:
                    if any(word in line.lower() for word in ['telefon', 'phone', 'tlf']):
                        phones = extract_phones(line)
                        if phones:
                            info['phone'] = phones[0]
                
                # Company name
                if 'name' not in info:
                    # Look for A/S, ApS, IVS suffixes
                    if any(suffix in line for suffix in [' A/S', ' ApS', ' A.S', ' IVS', ' I/S']):
                        if len(line) < 80 and not line.isupper():
                            info['name'] = line
    
    except Exception as e:
        logger.warning(f"Failed to extract company info: {e}")
    
    return info
