"""
Professional PDF Scraper - Camelot First
==========================================
Uses Camelot as primary extraction method with intelligent table parsing.
Properly identifies columns, filters headers, and validates data.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
import logging

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
# Enhanced Pattern Detection
# ============================================================================

def is_valid_person_name(text: str) -> bool:
    """
    Strict validation for person names.
    Must be 2-4 words, each capitalized, no numbers, no roles.
    """
    if not text or len(text) < 3 or len(text) > 60:
        return False
    
    # Remove common prefixes
    text = text.strip()
    
    # Must not be a role or company
    role_keywords = ['projektleder', 'kontaktperson', 'bygherre', 'entr', 'entreprenør', 
                     'rådgiver', 'ingeniør', 'chef', 'direktør', 'a/s', 'aps', 'totalentreprenør']
    if any(kw in text.lower() for kw in role_keywords):
        return False
    
    # Must not be all caps (headers)
    if text.isupper() and len(text) > 10:
        return False
    
    # Split into words
    words = text.split()
    
    # Must have 2-4 words
    if len(words) < 2 or len(words) > 4:
        return False
    
    # Each word should start with capital letter
    capitalized = [w for w in words if w and w[0].isupper() and w[0].isalpha()]
    if len(capitalized) < 2:
        return False
    
    # Should not contain numbers (except maybe Jr., II, III)
    if any(char.isdigit() for char in text):
        return False
    
    # Each word should be mostly letters
    for word in words:
        if len([c for c in word if c.isalpha()]) < len(word) * 0.7:
            return False
    
    return True

def extract_phones(text: str) -> List[str]:
    """Extract Danish phone numbers (8 digits)"""
    if not text:
        return []
    
    phones = []
    # Pattern for 8-digit Danish numbers with optional formatting
    pattern = re.compile(r'(?:\+45\s*)?(\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2})')
    
    for match in pattern.finditer(str(text)):
        phone = match.group(1).replace(' ', '').replace('-', '')
        if len(phone) == 8 and phone.isdigit():
            phones.append(phone)
    
    # Also try to find plain 8-digit numbers
    plain_pattern = re.compile(r'\b(\d{8})\b')
    for match in plain_pattern.finditer(str(text)):
        phone = match.group(1)
        if phone not in phones:
            phones.append(phone)
    
    return phones

def extract_emails(text: str) -> List[str]:
    """Extract email addresses"""
    if not text:
        return []
    
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return list(set(pattern.findall(str(text))))

def is_header_row(row: pd.Series) -> bool:
    """Check if a row is a table header"""
    row_text = ' '.join(str(cell).lower() for cell in row if pd.notna(cell))
    
    # Common header keywords
    header_keywords = [
        'navn', 'name', 'firma', 'company', 'telefon', 'phone', 'email', 'e-mail',
        'projektnavn', 'project', 'budget', 'region', 'stadie', 'dato', 'date',
        'kontakter', 'projekter', 'udbud', '#', 'nr.'
    ]
    
    # If contains multiple header keywords, it's likely a header
    keyword_count = sum(1 for kw in header_keywords if kw in row_text)
    if keyword_count >= 2:
        return True
    
    # If row has many empty cells, might be header
    empty_count = sum(1 for cell in row if pd.isna(cell) or str(cell).strip() == '')
    if empty_count > len(row) * 0.5:
        return True
    
    return False

def is_project_name(text: str) -> bool:
    """Check if text looks like an actual project name"""
    if not text or len(text) < 10:
        return False
    
    # Should not be just a stage
    stages = ['udførelsesproces', 'udbudsproces', 'projekteringsproces', 
              'planlægningsproces', 'afsluttet']
    if text.lower().strip() in stages:
        return False
    
    # Should not be just a trade
    trades = ['tømrer entr', 'el-entreprenør', 'beton entr', 'stål entr',
              'element', 'vindue', 'facade', 'gulventreprenører', 'jord/kloak',
              'fundering', 'råhusproducent', 'brandsikring', 'tagdækning',
              'belægning', 'snedker', 'vvs entr']
    if any(trade in text.lower() for trade in trades):
        return False
    
    # Should not be headers
    if text.upper() in ['PROJEKTER', 'KONTAKTER', 'UDBUD', 'STADIE']:
        return False
    
    # Should have some substance
    if len(text.strip()) > 15:
        return True
    
    return False

# ============================================================================
# Smart Column Identification
# ============================================================================

def identify_table_columns(df: pd.DataFrame) -> Dict[int, str]:
    """
    Intelligently identify what each column contains.
    Returns: {column_index: column_type}
    """
    
    column_map = {}
    
    for col_idx in range(len(df.columns)):
        col_data = df.iloc[:, col_idx]
        
        # Sample non-null values
        samples = [str(val) for val in col_data if pd.notna(val) and str(val).strip()]
        if not samples:
            column_map[col_idx] = 'empty'
            continue
        
        # Take first 20 samples
        samples = samples[:20]
        
        # Count patterns
        name_count = sum(1 for s in samples if is_valid_person_name(s))
        phone_count = sum(1 for s in samples if len(extract_phones(s)) > 0)
        email_count = sum(1 for s in samples if len(extract_emails(s)) > 0)
        number_count = sum(1 for s in samples if s.isdigit() and len(s) < 5)
        
        # Money pattern
        money_count = sum(1 for s in samples if any(kw in s.lower() for kw in ['mio', 'kr', 'dkk']))
        
        # Date pattern
        date_count = sum(1 for s in samples if re.search(r'(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)', s.lower()))
        
        # Region pattern
        region_count = sum(1 for s in samples if any(r in s for r in ['Hovedstaden', 'Sjælland', 'Midtjylland', 'Nordjylland', 'Syddanmark']))
        
        # Determine type
        total = len(samples)
        
        if phone_count / total > 0.4:
            column_map[col_idx] = 'phone'
        elif email_count / total > 0.3:
            column_map[col_idx] = 'email'
        elif name_count / total > 0.5:
            column_map[col_idx] = 'name'
        elif money_count / total > 0.4:
            column_map[col_idx] = 'budget'
        elif date_count / total > 0.4:
            column_map[col_idx] = 'date'
        elif region_count / total > 0.3:
            column_map[col_idx] = 'region'
        elif number_count / total > 0.5:
            column_map[col_idx] = 'id'
        else:
            # Check if it's long text (project names)
            avg_length = sum(len(s) for s in samples) / len(samples)
            if avg_length > 30:
                column_map[col_idx] = 'project_name'
            else:
                column_map[col_idx] = 'text'
    
    return column_map

# ============================================================================
# Contact Extraction
# ============================================================================

def extract_contacts_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract contacts from a properly identified contact table"""
    
    contacts = []
    column_map = identify_table_columns(df)
    
    logger.info(f"Contact table column map: {column_map}")
    
    # Find which columns are which
    name_cols = [i for i, t in column_map.items() if t == 'name']
    phone_cols = [i for i, t in column_map.items() if t == 'phone']
    email_cols = [i for i, t in column_map.items() if t == 'email']
    
    if not name_cols:
        logger.warning("No name column found in contact table")
        return []
    
    # Process each row
    for idx, row in df.iterrows():
        # Skip header rows
        if is_header_row(row):
            continue
        
        contact = {}
        
        # Extract name
        for name_col in name_cols:
            name = str(row.iloc[name_col]).strip()
            if pd.notna(row.iloc[name_col]) and is_valid_person_name(name):
                contact['name'] = name
                break
        
        if 'name' not in contact:
            continue  # Skip rows without valid names
        
        # Extract phone
        for phone_col in phone_cols:
            if pd.notna(row.iloc[phone_col]):
                phones = extract_phones(str(row.iloc[phone_col]))
                if phones:
                    contact['phone'] = phones[0]
                    break
        
        # Also check other columns for phones
        for i, cell in enumerate(row):
            if i not in phone_cols and pd.notna(cell):
                phones = extract_phones(str(cell))
                if phones and 'phone' not in contact:
                    contact['phone'] = phones[0]
        
        # Extract email
        for email_col in email_cols:
            if pd.notna(row.iloc[email_col]):
                emails = extract_emails(str(row.iloc[email_col]))
                if emails:
                    contact['email'] = emails[0]
                    break
        
        # Also check other columns for emails
        for i, cell in enumerate(row):
            if i not in email_cols and pd.notna(cell):
                emails = extract_emails(str(cell))
                if emails and 'email' not in contact:
                    contact['email'] = emails[0]
        
        # Must have at least phone or email
        if 'phone' in contact or 'email' in contact:
            contacts.append(contact)
    
    # Remove duplicates
    seen = set()
    unique_contacts = []
    for contact in contacts:
        key = (contact.get('name', ''), contact.get('phone', ''), contact.get('email', ''))
        if key not in seen:
            seen.add(key)
            unique_contacts.append(contact)
    
    logger.info(f"Extracted {len(unique_contacts)} unique contacts")
    return unique_contacts

# ============================================================================
# Project Extraction
# ============================================================================

def extract_projects_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract projects from a properly identified project table"""
    
    projects = []
    column_map = identify_table_columns(df)
    
    logger.info(f"Project table column map: {column_map}")
    
    # Find which columns are which
    name_cols = [i for i, t in column_map.items() if t == 'project_name']
    budget_cols = [i for i, t in column_map.items() if t == 'budget']
    date_cols = [i for i, t in column_map.items() if t == 'date']
    region_cols = [i for i, t in column_map.items() if t == 'region']
    
    # Process each row
    for idx, row in df.iterrows():
        # Skip header rows
        if is_header_row(row):
            continue
        
        project = {}
        
        # Extract project name
        for name_col in name_cols:
            if pd.notna(row.iloc[name_col]):
                name = str(row.iloc[name_col]).strip()
                if is_project_name(name):
                    project['name'] = name
                    break
        
        # If no name column, look for long text in any column
        if 'name' not in project:
            for i, cell in enumerate(row):
                if pd.notna(cell):
                    text = str(cell).strip()
                    if is_project_name(text):
                        project['name'] = text
                        break
        
        if 'name' not in project:
            continue  # Skip rows without project names
        
        # Extract budget
        for budget_col in budget_cols:
            if pd.notna(row.iloc[budget_col]):
                budget = str(row.iloc[budget_col]).strip()
                if any(kw in budget.lower() for kw in ['mio', 'mia', 'kr']):
                    project['budget'] = budget
                    break
        
        # Extract dates
        dates = []
        for date_col in date_cols:
            if pd.notna(row.iloc[date_col]):
                dates.append(str(row.iloc[date_col]).strip())
        if dates:
            project['dates'] = dates
        
        # Extract region
        for region_col in region_cols:
            if pd.notna(row.iloc[region_col]):
                project['region'] = str(row.iloc[region_col]).strip()
                break
        
        # Add project if it has meaningful data
        if len(project) > 1:  # More than just name
            projects.append(project)
    
    # Remove duplicates
    seen = set()
    unique_projects = []
    for project in projects:
        key = (project.get('name', ''), project.get('budget', ''))
        if key not in seen:
            seen.add(key)
            unique_projects.append(project)
    
    logger.info(f"Extracted {len(unique_projects)} unique projects")
    return unique_projects

# ============================================================================
# Table Classification
# ============================================================================

def classify_table_smart(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Classify table type with better accuracy.
    Returns: (type, confidence)
    """
    
    if df.empty or len(df) < 2:
        return ('unknown', 0.0)
    
    column_map = identify_table_columns(df)
    
    # Count column types
    name_count = sum(1 for t in column_map.values() if t == 'name')
    phone_count = sum(1 for t in column_map.values() if t == 'phone')
    email_count = sum(1 for t in column_map.values() if t == 'email')
    project_name_count = sum(1 for t in column_map.values() if t == 'project_name')
    budget_count = sum(1 for t in column_map.values() if t == 'budget')
    date_count = sum(1 for t in column_map.values() if t == 'date')
    
    # Contact table scoring
    contact_score = 0.0
    if name_count >= 1:
        contact_score += 3.0
    if phone_count >= 1:
        contact_score += 3.0
    if email_count >= 1:
        contact_score += 2.0
    
    # Project table scoring
    project_score = 0.0
    if project_name_count >= 1:
        project_score += 4.0
    elif name_count >= 1:
        project_score += 1.0
    if budget_count >= 1:
        project_score += 3.0
    if date_count >= 1:
        project_score += 2.0
    
    # Determine type
    max_score = max(contact_score, project_score)
    
    if max_score == 0:
        return ('unknown', 0.0)
    
    confidence = min(max_score / 8.0, 1.0)
    
    if contact_score > project_score and contact_score >= 4.0:
        return ('contact', confidence)
    elif project_score > contact_score and project_score >= 4.0:
        return ('project', confidence)
    else:
        return ('unknown', confidence)

# ============================================================================
# Main Parser
# ============================================================================

def parse_pdf(pdf_path: str) -> Dict:
    """
    Professional PDF parsing with Camelot-first approach.
    """
    
    logger.info(f"Parsing PDF: {pdf_path}")
    
    # Extract company info
    company_info = extract_company_info(pdf_path)
    
    # Extract tables with Camelot (both methods)
    all_tables = []
    
    # Method 1: Lattice (tables with borders)
    try:
        logger.info("Extracting with Camelot lattice...")
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice', strip_text='\n')
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 2:
                all_tables.append((table.df, table.page, 'lattice', table.parsing_report.get('accuracy', 0)))
        logger.info(f"Found {len(tables)} lattice tables")
    except Exception as e:
        logger.warning(f"Camelot lattice failed: {e}")
    
    # Method 2: Stream (tables without clear borders)
    try:
        logger.info("Extracting with Camelot stream...")
        tables = camelot.read_pdf(
            pdf_path, 
            pages='all', 
            flavor='stream',
            edge_tol=100,
            row_tol=15,
            column_tol=10
        )
        for table in tables:
            if not table.df.empty and table.df.shape[0] > 2:
                # Check for duplicates
                is_dup = False
                for existing_df, _, _, _ in all_tables:
                    if existing_df.shape == table.df.shape:
                        is_dup = True
                        break
                if not is_dup:
                    all_tables.append((table.df, table.page, 'stream', table.parsing_report.get('accuracy', 0)))
        logger.info(f"Found {len(tables)} stream tables")
    except Exception as e:
        logger.warning(f"Camelot stream failed: {e}")
    
    logger.info(f"Total tables to process: {len(all_tables)}")
    
    # Process tables
    contacts = []
    projects = []
    quality_scores = []
    
    for df, page, method, accuracy in all_tables:
        # Classify table
        table_type, confidence = classify_table_smart(df)
        
        logger.info(f"Page {page}, {method}: {table_type} (confidence: {confidence:.2f}, accuracy: {accuracy:.0f}%)")
        
        if table_type == 'unknown' or confidence < 0.4:
            continue
        
        quality_scores.append(confidence)
        
        # Extract data
        if table_type == 'contact':
            new_contacts = extract_contacts_from_table(df)
            contacts.extend(new_contacts)
        elif table_type == 'project':
            new_projects = extract_projects_from_table(df)
            projects.extend(new_projects)
    
    # Calculate quality
    avg_confidence = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    logger.info(f"Final: {len(contacts)} contacts, {len(projects)} projects")
    
    return {
        'company_info': company_info,
        'contacts': contacts,
        'projects': projects,
        'tenders': [],
        'quality': {
            'avg_confidence': round(avg_confidence, 2),
            'tables_processed': len(quality_scores),
            'extraction_method': 'camelot-professional'
        },
        'summary': {
            'contacts': len(contacts),
            'projects': len(projects),
            'tenders': 0
        }
    }

# ============================================================================
# Company Info (same as before)
# ============================================================================

def extract_company_info(pdf_path: str) -> Dict[str, str]:
    """Extract company info from first page"""
    info = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return info
            
            text = pdf.pages[0].extract_text() or ""
            lines = text.split('\n')[:30]
            
            for line in lines:
                line = line.strip()
                
                if 'cvr' in line.lower():
                    match = re.search(r'\b(\d{8})\b', line)
                    if match:
                        info['cvr'] = match.group(1)
                
                emails = extract_emails(line)
                if emails and 'email' not in info:
                    info['email'] = emails[0]
                
                if 'http' in line.lower():
                    match = re.search(r'https?://[^\s]+', line)
                    if match:
                        info['website'] = match.group(0)
                
                if 'phone' not in info and 'telefon' in line.lower():
                    phones = extract_phones(line)
                    if phones:
                        info['phone'] = phones[0]
                
                if 'name' not in info:
                    if any(suffix in line for suffix in [' A/S', ' ApS', ' A.S', ' IVS']):
                        if len(line) < 60 and not line.isupper():
                            info['name'] = line
    except:
        pass
    
    return info
