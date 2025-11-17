"""
Enhanced Professional PDF Scraper for Byggefakta (New & Old Layouts)
=====================================================================
FIXED VERSION with improved role extraction for English and Danish roles
- Separates "Project leader" roles from "Handled" roles
- Filters out uninteresting roles like "Purchasers"

Key improvements:
- Case-insensitive column detection
- Two-tier role structure (project_roles vs handled_roles)
- Direct text extraction from role columns
- Multiple role parsing strategies
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

def fix_camelcase_boundaries(text: str) -> str:
    """Fix missing spaces in camelCase text"""
    if not text or len(text) < 2:
        return text
    
    result = []
    for i, char in enumerate(text):
        if (i > 0 and 
            char.isupper() and 
            text[i-1].islower() and
            (i == 1 or text[i-2] != ' ')):
            result.append(' ')
        result.append(char)
    
    return ''.join(result)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_multiline(text: str) -> str:
    """Clean multi-line cell text but preserve meaningful line breaks"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

# ============================================================================
# Pattern Detection
# ============================================================================

def is_valid_person_name(text: str) -> bool:
    """Validate person names"""
    if not text or len(text) < 3 or len(text) > 70:
        return False
    
    text = clean_text(text)
    
    blacklist = [
        'projekt', 'kontakt', 'entr', 'entrepren', 'rådgiver', 'ingeniør',
        'chef', 'direktør', 'a/s', 'aps', 'firma', 'rolle', 'telefon',
        'navn', 'cvr', 'total', 'hoved', 'bygge', 'element', 'beton',
        'tømrer', 'snedker', 'murer', 'maler', 'elektriker', 'vvs',
        'tagdækning', 'facade', 'gulv', 'vindue', 'dør', 'stål', 'smede',
        'projektleder', 'byggeleder', 'sagsansvarlig', 'projektchef',
        'handled', 'project', 'leader', 'contractor', 'producer'
    ]
    
    text_lower = text.lower()
    if any(word in text_lower for word in blacklist):
        return False
    
    if text.isupper() and len(text) > 8:
        return False
    
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False
    
    capitalized = [w for w in words if w and len(w) > 0 and w[0].isupper()]
    if len(capitalized) < 2:
        return False
    
    if any(char.isdigit() for char in text):
        return False
    
    for word in words:
        if len(word) > 1:
            letters = sum(1 for c in word if c.isalpha())
            if letters < len(word) * 0.7:
                return False
    
    return True

def extract_phones(text: str) -> List[str]:
    """Extract Danish phone numbers"""
    if not text or pd.isna(text):
        return []
    
    text = str(text)
    phones = []
    
    patterns = [
        r'(?:\+45\s*)?(\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2})',
        r'\b(\d{8})\b'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            phone = re.sub(r'[\s\-]', '', match.group(1))
            if len(phone) == 8 and phone.isdigit():
                if not re.search(r'(19|20)\d{2}', phone):
                    context = text[max(0, match.start()-10):match.end()+10].lower()
                    if 'cvr' not in context and 'org nr' not in context:
                        phones.append(phone)
    
    return list(dict.fromkeys(phones))

def extract_emails(text: str) -> List[str]:
    """Extract email addresses"""
    if not text or pd.isna(text):
        return []
    
    pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return list(set(pattern.findall(str(text))))

def extract_roles_from_text(text: str) -> Dict[str, List[str]]:
    """
    IMPROVED: Extract and categorize roles into project roles vs handled roles
    
    Returns dict with:
    - 'project_roles': Important roles (Project leader, Project manager, etc.)
    - 'handled_roles': Support/involvement roles (Handled. [role])
    
    Handles formats like:
    - "Handled. Steel contractor"
    - "Project leader. Total contractor"
    - "Projektleder. Totalentreprenør"
    - Multiple roles separated by newlines or periods
    """
    if not text or pd.isna(text):
        return {'project_roles': [], 'handled_roles': []}
    
    text = clean_multiline(text)
    project_roles = []
    handled_roles = []
    
    # Skip uninteresting roles
    uninteresting = ['purchaser', 'purchasers', 'indkøber', 'indkøbere']
    
    # Prefixes that indicate project roles (important)
    project_prefixes = [
        'project leader', 'projektleder', 'project manager', 'projektchef',
        'project planning leader', 'production manager', 'head of project',
        'byggeleder', 'sagsansvarlig', 'projekteringsleder'
    ]
    
    # Handled prefix
    handled_prefix = 'handled'
    
    # Strategy 1: Parse line by line and look for prefixes
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip names, phones, emails
        if is_valid_person_name(line):
            continue
        if extract_phones(line) or extract_emails(line):
            continue
        
        line_lower = line.lower()
        
        # Check if line starts with a project prefix
        is_project_role_line = any(
            line_lower.startswith(prefix) 
            for prefix in project_prefixes
        )
        
        # Check if line starts with "Handled"
        is_handled_line = line_lower.startswith(handled_prefix)
        
        # Split by periods to extract individual roles
        segments = [s.strip() for s in line.split('.') if s.strip()]
        
        for i, segment in enumerate(segments):
            segment_lower = segment.lower()
            
            # Skip the prefix itself
            if segment_lower in project_prefixes + [handled_prefix]:
                continue
            
            # Skip uninteresting roles
            if any(unint in segment_lower for unint in uninteresting):
                continue
            
            # Must look like a role
            if not (3 < len(segment) < 100):
                continue
            
            # Check if it contains role keywords
            role_indicators = [
                # English
                'contractor', 'leader', 'manager', 'engineer', 
                'director', 'coordinator', 'consultant', 'architect',
                'supervisor', 'chief', 'specialist', 'producer', 'delivery',
                'planner', 'designer', 'supplier',
                # Trades/contractors (English & Danish)
                'carpenter', 'tømrer', 'snedker',
                'electrician', 'elektriker',
                'plumber', 'vvs',
                'mason', 'bricklayer', 'murer',
                'painter', 'maler',
                'roofer', 'tagger', 'tagdækker',
                'blacksmith', 'smed', 'smede',
                'glazier', 'window', 'vindue',
                'flooring', 'gulv',
                'facade', 'facadist',
                'steel', 'stål',
                'concrete', 'beton',
                'landscape', 'anlæg',
                'excavation', 'grave',
                'tile', 'flise',
                # Danish
                'entreprenør', 'leder', 'chef', 'ingeniør', 'rådgiver',
                'producent', 'levering', 'leverandør'
            ]
            
            if not any(indicator in segment_lower for indicator in role_indicators):
                continue
            
            # Clean up the role
            role = re.sub(r'\s+', ' ', segment).strip()
            
            # Determine if this is a project role or handled role
            # Check the previous segment to see if it was a prefix
            if i > 0:
                prev_segment = segments[i-1].lower().strip()
                if prev_segment in project_prefixes or any(p in prev_segment for p in project_prefixes):
                    if role and role not in project_roles:
                        project_roles.append(role)
                    continue
                elif prev_segment == handled_prefix or handled_prefix in prev_segment:
                    if role and role not in handled_roles:
                        handled_roles.append(role)
                    continue
            
            # Or check if the role line itself started with a prefix
            if is_project_role_line:
                if role and role not in project_roles:
                    project_roles.append(role)
            elif is_handled_line:
                if role and role not in handled_roles:
                    handled_roles.append(role)
            else:
                # Default: if it contains "leader" or "manager", it's a project role
                if any(kw in segment_lower for kw in ['leader', 'leder', 'manager', 'chef', 'head']):
                    if role and role not in project_roles:
                        project_roles.append(role)
                else:
                    # Otherwise it's a handled role
                    if role and role not in handled_roles:
                        handled_roles.append(role)
    
    # Strategy 2: Regex patterns for Danish roles (if Strategy 1 didn't work)
    if not project_roles and not handled_roles:
        danish_project_patterns = [
            r'Projektleder[^.\n]*',
            r'Byggeleder[^.\n]*',
            r'Sagsansvarlig[^.\n]*',
            r'Projektchef[^.\n]*',
            r'Projekteringsleder[^.\n]*',
        ]
        
        for pattern in danish_project_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                role = clean_text(match)
                if role and role not in project_roles:
                    project_roles.append(role)
        
        danish_contractor_patterns = [
            r'Totalentreprenør',
            r'Hovedentreprenør',
            r'[A-ZÆØÅ][a-zæøå]+entreprenør',
        ]
        
        for pattern in danish_contractor_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                role = clean_text(match)
                if role and role not in handled_roles:
                    handled_roles.append(role)
    
    return {
        'project_roles': project_roles[:5],  # Max 5 project roles
        'handled_roles': handled_roles[:10]  # Max 10 handled roles
    }

# ============================================================================
# Table Classification
# ============================================================================

def detect_table_type(df: pd.DataFrame) -> Tuple[str, float]:
    """Detect if table contains contacts, projects, or tenders"""
    
    if df.empty or len(df) < 2:
        return ('unknown', 0.0)
    
    all_text = ' '.join(
        str(cell).lower() 
        for row in df.values 
        for cell in row 
        if pd.notna(cell)
    )
    
    # Contact table indicators
    contact_score = 0.0
    if 'navn' in all_text or 'name' in all_text:
        contact_score += 2.0
    if any(word in all_text for word in ['telefon', 'phone', 'mobil', 'tlf']):
        contact_score += 3.0
    if 'email' in all_text or 'e-mail' in all_text or 'mail' in all_text:
        contact_score += 2.0
    if 'rolle' in all_text or 'role' in all_text or 'kontaktperson' in all_text or 'projektleder' in all_text:
        contact_score += 2.0
    if 'firma' in all_text or 'company' in all_text:
        contact_score += 1.0
    
    name_count = 0
    for row in df.values[:20]:
        for cell in row:
            if pd.notna(cell) and is_valid_person_name(str(cell)):
                name_count += 1
    
    if name_count >= 5:
        contact_score += 3.0
    elif name_count >= 2:
        contact_score += 1.0
    
    # Project table indicators
    project_score = 0.0
    if 'projekt' in all_text:
        project_score += 3.0
    if any(word in all_text for word in ['budget', 'mio', 'kr', 'kr.']):
        project_score += 3.0
    if any(word in all_text for word in ['byggestart', 'dato', 'date', 'start']):
        project_score += 2.0
    if 'region' in all_text or 'hovedstaden' in all_text:
        project_score += 2.0
    if any(word in all_text for word in ['stadie', 'udførelse', 'stage']):
        project_score += 2.0
    if 'bæredygtighed' in all_text or 'sustainability' in all_text:
        project_score += 2.0
    if 'seneste' in all_text and 'opdatering' in all_text:
        project_score += 1.5
    if 'roller' in all_text and 'projekt' in all_text:
        project_score += 1.0
    
    # Tender table indicators
    tender_score = 0.0
    if 'udbud' in all_text:
        tender_score += 5.0
    if 'licitation' in all_text:
        tender_score += 3.0
    if all_text.count('arkiv') >= 3:
        tender_score += 2.0
    if 'udbudsrolle' in all_text:
        tender_score += 2.0
    
    scores = {'contact': contact_score, 'project': project_score, 'tender': tender_score}
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]
    
    if max_score < 3.0:
        return ('unknown', 0.0)
    
    confidence = min(max_score / 10.0, 1.0)
    return (max_type, confidence)

# ============================================================================
# Contact Extraction (IMPROVED)
# ============================================================================

def find_column_indices(df: pd.DataFrame, keywords: List[str]) -> List[int]:
    """FIXED: Find column indices with case-insensitive matching"""
    indices = []
    
    for col_idx in range(len(df.columns)):
        col_text = ' '.join(
            str(df.iloc[i, col_idx]).lower() 
            for i in range(min(5, len(df))) 
            if pd.notna(df.iloc[i, col_idx])
        )
        
        # Case-insensitive matching
        if any(kw.lower() in col_text for kw in keywords):
            indices.append(col_idx)
            logger.info(f"Found column {col_idx} for keywords {keywords}")
    
    return indices

def detect_id_column(df: pd.DataFrame) -> Optional[int]:
    """Detect if table has a number/ID column"""
    for col_idx in range(min(3, len(df.columns))):
        numbers = []
        for i in range(min(10, len(df))):
            cell = str(df.iloc[i, col_idx]).strip()
            if cell.isdigit() and len(cell) <= 3:
                numbers.append(int(cell))
        
        if len(numbers) >= 3:
            sorted_nums = sorted(numbers)
            if sorted_nums[-1] - sorted_nums[0] <= len(numbers) * 2:
                logger.info(f"Detected ID column (#) at index {col_idx}")
                return col_idx
    
    return None

def merge_multirow_entries(df: pd.DataFrame, boundary_cols: List[int]) -> pd.DataFrame:
    """Merge multi-row entries into single rows"""
    if df.empty or not boundary_cols:
        return df
    
    id_col = detect_id_column(df)
    primary_col = boundary_cols[0]
    
    merged_rows = []
    current_entry = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        is_new_entry = False
        
        if id_col is not None:
            cell_val = str(row.iloc[id_col]).strip()
            if cell_val.isdigit():
                is_new_entry = True
        else:
            cell = str(row.iloc[primary_col]) if primary_col < len(row) else ""
            is_new_entry = cell.strip() and cell not in ['', 'nan', 'None']
        
        if is_new_entry:
            if current_entry is not None:
                merged_rows.append(current_entry)
            current_entry = row.copy()
        else:
            if current_entry is not None:
                for col_idx in range(len(row)):
                    cell_value = str(row.iloc[col_idx]).strip()
                    if cell_value and cell_value not in ['', 'nan', 'None']:
                        existing = str(current_entry.iloc[col_idx]).strip()
                        if not existing or existing in ['', 'nan', 'None']:
                            current_entry.iloc[col_idx] = cell_value
                        else:
                            current_entry.iloc[col_idx] = existing + '\n' + cell_value
    
    if current_entry is not None:
        merged_rows.append(current_entry)
    
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        logger.info(f"Merged {len(df)} rows into {len(merged_df)} entries")
        return merged_df
    
    return df

def extract_contacts_from_table(df: pd.DataFrame) -> List[Dict]:
    """
    IMPROVED: Extract contacts with categorized role extraction
    """
    
    contacts = []
    
    logger.info(f"Extracting contacts from table with shape {df.shape}")
    
    # Find relevant columns (case-insensitive now)
    name_cols = find_column_indices(df, ['navn', 'name'])
    phone_cols = find_column_indices(df, ['telefon', 'phone', 'mobil', 'phones'])
    email_cols = find_column_indices(df, ['email', 'e-mail', 'mail'])
    role_cols = find_column_indices(df, ['rolle', 'role', 'position', 'titel', 'title'])
    
    logger.info(f"Column indices - names: {name_cols}, phones: {phone_cols}, emails: {email_cols}, roles: {role_cols}")
    
    # If no explicit columns found, try to infer
    if not name_cols:
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
                logger.info(f"Inferred name column: {best_col}")
    
    if not name_cols:
        logger.warning("No name column found in contact table")
        return []
    
    # Merge multi-row contacts
    df = merge_multirow_entries(df, name_cols)
    
    # Detect ID column
    id_col = detect_id_column(df)
    
    # Skip header rows
    start_row = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if any(kw in row_text for kw in ['navn', 'name', 'firma', 'telefon', 'rolle', 'role']):
            start_row = i + 1
            logger.info(f"Skipping header rows, starting at row {start_row}")
    
    # Extract contacts
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        
        contact = {}
        
        # Extract ID
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
        
        # Extract phones (collect ALL)
        all_phones = []
        for col_idx in phone_cols + list(range(len(row))):
            if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                phones = extract_phones(str(row.iloc[col_idx]))
                if phones:
                    all_phones.extend(phones)
        
        if all_phones:
            seen = set()
            unique_phones = []
            for phone in all_phones:
                if phone not in seen:
                    seen.add(phone)
                    unique_phones.append(phone)
            
            if len(unique_phones) == 1:
                contact['phone'] = unique_phones[0]
            else:
                contact['phones'] = unique_phones
                contact['phone'] = unique_phones[0]
        
        # Extract emails
        all_emails = []
        for col_idx in email_cols + list(range(len(row))):
            if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                emails = extract_emails(str(row.iloc[col_idx]))
                if emails:
                    all_emails.extend(emails)
        
        if all_emails:
            unique_emails = list(dict.fromkeys(all_emails))
            if len(unique_emails) == 1:
                contact['email'] = unique_emails[0]
            else:
                contact['emails'] = unique_emails
                contact['email'] = unique_emails[0]
        
        # IMPROVED: Extract categorized roles
        all_project_roles = []
        all_handled_roles = []
        
        # Strategy 1: If we found role columns, extract from those first
        if role_cols:
            for col_idx in role_cols:
                if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                    cell_text = str(row.iloc[col_idx])
                    # Direct extraction from role column
                    roles_dict = extract_roles_from_text(cell_text)
                    all_project_roles.extend(roles_dict['project_roles'])
                    all_handled_roles.extend(roles_dict['handled_roles'])
                    logger.debug(f"Extracted from column {col_idx}: {roles_dict}")
        
        # Strategy 2: Check all other columns (except name columns) for role-like text
        if not all_project_roles and not all_handled_roles:
            for col_idx in range(len(row)):
                if col_idx not in name_cols and col_idx not in phone_cols and pd.notna(row.iloc[col_idx]):
                    cell_text = str(row.iloc[col_idx])
                    roles_dict = extract_roles_from_text(cell_text)
                    if roles_dict['project_roles'] or roles_dict['handled_roles']:
                        all_project_roles.extend(roles_dict['project_roles'])
                        all_handled_roles.extend(roles_dict['handled_roles'])
                        logger.debug(f"Extracted from column {col_idx}: {roles_dict}")
        
        # Remove duplicates
        if all_project_roles:
            contact['project_roles'] = list(dict.fromkeys(all_project_roles))[:5]
        if all_handled_roles:
            contact['handled_roles'] = list(dict.fromkeys(all_handled_roles))[:10]
        
        if contact.get('project_roles') or contact.get('handled_roles'):
            logger.debug(f"Contact {contact['name']} - Project: {contact.get('project_roles')}, Handled: {contact.get('handled_roles')}")
        
        # Include contact if has name + (phone OR email OR role)
        if 'phone' in contact or 'email' in contact or 'project_roles' in contact or 'handled_roles' in contact:
            contacts.append(contact)
        else:
            logger.debug(f"Skipping contact {contact.get('name')} - no phone/email/role")
    
    # Deduplicate
    seen = set()
    unique = []
    for contact in contacts:
        key = (contact.get('name', ''), contact.get('phone', ''), contact.get('email', ''))
        if key not in seen:
            seen.add(key)
            unique.append(contact)
    
    logger.info(f"Extracted {len(unique)} unique contacts")
    
    # Log role extraction statistics
    contacts_with_project = sum(1 for c in unique if 'project_roles' in c)
    contacts_with_handled = sum(1 for c in unique if 'handled_roles' in c)
    logger.info(f"Contacts with project roles: {contacts_with_project}/{len(unique)}, handled roles: {contacts_with_handled}/{len(unique)}")
    
    return unique

# ============================================================================
# Project Extraction
# ============================================================================

def extract_budget(text: str) -> Optional[str]:
    """Extract budget value from text"""
    if not text or pd.isna(text):
        return None
    
    text = clean_multiline(text)
    
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
    
    month_pattern = r'(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)[a-z]*\.?\s+\d{4}'
    match = re.search(month_pattern, text, re.IGNORECASE)
    if match:
        return clean_text(match.group(0))
    
    day_month_year = r'\d{1,2}\s+(?:jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)[a-z]*\.?\s+\d{4}'
    match = re.search(day_month_year, text, re.IGNORECASE)
    if match:
        return clean_text(match.group(0))
    
    date_pattern = r'\d{1,2}[-./]\d{1,2}[-./]\d{4}'
    match = re.search(date_pattern, text)
    if match:
        return clean_text(match.group(0))
    
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
        'Skitseprojekt',
        'Construction',
        'Procurement',
        'Projecting'
    ]
    
    text_clean = clean_multiline(text)
    
    for stage in stages:
        if stage.lower() in text_clean.lower():
            return stage
    
    return None

def extract_projects_from_table(df: pd.DataFrame) -> List[Dict]:
    """Extract projects with all details including categorized roles"""
    
    projects = []
    
    logger.info(f"Extracting projects from table with shape {df.shape}")
    
    # Skip header rows
    start_row = 0
    for i in range(min(10, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if any(kw in row_text for kw in ['projekt', 'budget', 'region', 'rolle', 'byggestart']):
            start_row = i + 1
    
    df_data = df.iloc[start_row:].copy() if start_row < len(df) else df.copy()
    df_data = df_data.reset_index(drop=True)
    
    # Merge multi-row projects
    df_data = merge_multirow_entries(df_data, [0])
    
    id_col = detect_id_column(df_data)
    
    for idx in range(len(df_data)):
        row = df_data.iloc[idx]
        
        project = {}
        
        # Extract ID
        if id_col is not None and id_col < len(row):
            project_id = str(row.iloc[id_col]).strip().split('\n')[0].strip()
            if project_id.isdigit():
                project['id'] = project_id
        
        # Collect all non-empty cells
        cells = [clean_multiline(str(cell)) for cell in row if pd.notna(cell) and str(cell).strip()]
        
        if not cells:
            continue
        
        # Extract project name
        project_name = None
        for cell in cells:
            if len(cell) > 15 and not re.match(r'^\d+\s+(mio|mia)', cell.lower()):
                if not any(word in cell.lower() for word in ['hovedstaden', 'sjælland', 'entr.', 'totalentreprenør']):
                    if not re.match(r'^\d{1,2}\s+\w+\.?\s+\d{4}', cell):
                        project_name = cell
                        break
        
        if not project_name:
            for cell in cells:
                if len(cell) > 10:
                    project_name = cell
                    break
        
        if not project_name:
            continue
        
        # Clean project name
        project_name = re.sub(r'\s*\n\s*', ' ', project_name)
        project_name = re.sub(r'\s+', ' ', project_name).strip()
        project_name = fix_camelcase_boundaries(project_name)
        
        project['name'] = project_name
        
        # Extract other fields
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
        
        # Extract last updated date
        update_date = None
        for cell in cells:
            if 'byggestart' not in cell.lower():
                date_match = re.search(
                    r'\d{1,2}\s+(?:jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)[a-z]*\.?\s+\d{4}',
                    cell,
                    re.IGNORECASE
                )
                if date_match:
                    potential_date = clean_text(date_match.group(0))
                    update_date = potential_date
        
        if update_date:
            project['last_updated'] = update_date
        
        # Extract categorized roles
        roles_dict = extract_roles_from_text(all_text)
        if roles_dict['project_roles']:
            project['project_roles'] = roles_dict['project_roles'][:3]
        if roles_dict['handled_roles']:
            project['handled_roles'] = roles_dict['handled_roles'][:5]
        
        # Check for sustainability
        if '✓' in all_text or 'bæredygtighed' in all_text.lower():
            project['sustainability'] = True
        
        if len(project) >= 2:
            projects.append(project)
    
    # Deduplicate
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
    
    start_row = 0
    for i in range(min(5, len(df))):
        row_text = ' '.join(str(cell).lower() for cell in df.iloc[i] if pd.notna(cell))
        if 'udbud' in row_text or 'licitation' in row_text:
            start_row = i + 1
    
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
        
        # Extract trade/type using improved function
        all_text = ' '.join(cells)
        roles_dict = extract_roles_from_text(all_text)
        
        # For tenders, we mostly care about the contractor type
        if roles_dict['handled_roles']:
            tender['trade'] = roles_dict['handled_roles'][0]
        elif roles_dict['project_roles']:
            tender['trade'] = roles_dict['project_roles'][0]
        
        # Extract dates
        date = extract_date(all_text)
        if date:
            tender['date'] = date
        
        # Mark as archived
        if 'arkiv' in all_text.lower():
            tender['status'] = 'Archived'
        
        if 'name' in tender:
            tenders.append(tender)
    
    logger.info(f"Extracted {len(tenders)} tenders")
    return tenders

# ============================================================================
# pdfplumber Table Extraction
# ============================================================================

def extract_tables_with_pdfplumber(pdf_path: str) -> List[Tuple[pd.DataFrame, int, str, float]]:
    """Extract tables using pdfplumber"""
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if not table or len(table) < 2:
                        continue
                    
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                    except:
                        df = pd.DataFrame(table)
                    
                    df = df.replace('', np.nan)
                    df = df.replace(None, np.nan)
                    df = df.dropna(how='all')
                    df = df.dropna(axis=1, how='all')
                    
                    if not df.empty and df.shape[0] > 1:
                        tables.append((df, page_num, 'pdfplumber', 0.8))
                        logger.info(f"pdfplumber: Page {page_num}, shape {df.shape}")
        
        logger.info(f"pdfplumber extracted {len(tables)} tables")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    return tables

# ============================================================================
# Text-based Fallback Extraction
# ============================================================================

def extract_from_text_fallback(pdf_path: str) -> Dict:
    """Fallback: Extract contacts and projects from text"""
    contacts = []
    projects = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                
                # Extract contacts
                if 'KONTAKTER' in text or 'CONTACTS' in text or 'Kontakter' in text:
                    lines = text.split('\n')
                    in_contact_section = False
                    current_contact = None
                    
                    for i, line in enumerate(lines):
                        line_clean = line.strip()
                        
                        if any(kw in line for kw in ['KONTAKTER', 'CONTACTS', 'Kontakter']):
                            in_contact_section = True
                            continue
                        
                        if in_contact_section and any(header in line for header in ['PROJEKTER', 'PROJECTS', 'Projekter', 'OPLYSNINGER', 'Hubexo', 'UDBUD', 'Udbud']):
                            if current_contact:
                                contacts.append(current_contact)
                            break
                        
                        if in_contact_section:
                            if 'Navn' in line and 'Telefon' in line:
                                continue
                            
                            phones_in_line = extract_phones(line)
                            if phones_in_line:
                                words = line_clean.split()
                                
                                name_candidates = []
                                for j in range(len(words) - 1):
                                    if j < len(words) - 1:
                                        two_word = ' '.join(words[j:j+2])
                                        if is_valid_person_name(two_word):
                                            name_candidates.append(two_word)
                                    
                                    if j < len(words) - 2:
                                        three_word = ' '.join(words[j:j+3])
                                        if is_valid_person_name(three_word):
                                            name_candidates.append(three_word)
                                
                                if name_candidates:
                                    if current_contact:
                                        contacts.append(current_contact)
                                    
                                    best_name = max(name_candidates, key=len)
                                    current_contact = {'name': best_name}
                                    
                                    if len(phones_in_line) == 1:
                                        current_contact['phone'] = phones_in_line[0]
                                    else:
                                        current_contact['phones'] = phones_in_line
                                        current_contact['phone'] = phones_in_line[0]
                                    
                                    emails = extract_emails(line)
                                    if emails:
                                        current_contact['email'] = emails[0]
                                    
                                    # IMPROVED: Use new categorized role extraction
                                    roles_dict = extract_roles_from_text(line)
                                    if roles_dict['project_roles']:
                                        current_contact['project_roles'] = roles_dict['project_roles']
                                    if roles_dict['handled_roles']:
                                        current_contact['handled_roles'] = roles_dict['handled_roles']
                            
                            elif is_valid_person_name(line_clean):
                                if current_contact:
                                    contacts.append(current_contact)
                                
                                current_contact = {'name': line_clean}
                            
                            elif current_contact:
                                phones = extract_phones(line)
                                if phones and 'phone' not in current_contact:
                                    if len(phones) == 1:
                                        current_contact['phone'] = phones[0]
                                    else:
                                        current_contact['phones'] = phones
                                        current_contact['phone'] = phones[0]
                                
                                emails = extract_emails(line)
                                if emails and 'email' not in current_contact:
                                    current_contact['email'] = emails[0]
                                
                                # IMPROVED: Use new categorized role extraction
                                roles_dict = extract_roles_from_text(line)
                                if roles_dict['project_roles']:
                                    if 'project_roles' not in current_contact:
                                        current_contact['project_roles'] = []
                                    current_contact['project_roles'].extend(roles_dict['project_roles'])
                                if roles_dict['handled_roles']:
                                    if 'handled_roles' not in current_contact:
                                        current_contact['handled_roles'] = []
                                    current_contact['handled_roles'].extend(roles_dict['handled_roles'])
                    
                    if current_contact:
                        contacts.append(current_contact)
                
                # Extract projects
                if 'PROJEKTER' in text or 'PROJECTS' in text or 'Projekter' in text:
                    lines = text.split('\n')
                    in_project_section = False
                    
                    for line in lines:
                        if any(kw in line for kw in ['PROJEKTER', 'PROJECTS', 'Projekter']):
                            in_project_section = True
                            continue
                        
                        if in_project_section and any(header in line for header in ['KONTAKTER', 'CONTACTS', 'Kontakter', 'OPLYSNINGER', 'UDBUD', 'Udbud']):
                            break
                        
                        if in_project_section:
                            if extract_budget(line) or any(kw in line.lower() for kw in ['opførelse', 'renovering', 'ombygning', 'etablering']):
                                line_fixed = fix_camelcase_boundaries(line)
                                
                                project = {}
                                
                                parts = line_fixed.split()
                                name_parts = []
                                for part in parts:
                                    if (not re.match(r'^\d+$', part) and 
                                        not any(x in part.lower() for x in ['mio', 'mia', 'hovedstaden', 'entr', 'kr.']) and
                                        len(part) > 2):
                                        name_parts.append(part)
                                    else:
                                        if name_parts:
                                            break
                                
                                if name_parts:
                                    project['name'] = ' '.join(name_parts[:15])
                                    
                                    budget = extract_budget(line)
                                    if budget:
                                        project['budget'] = budget
                                    
                                    date = extract_date(line)
                                    if date:
                                        project['start_date'] = date
                                    
                                    region = extract_region(line)
                                    if region:
                                        project['region'] = region
                                    
                                    stage = extract_stage(line)
                                    if stage:
                                        project['stage'] = stage
                                    
                                    # IMPROVED: Use new categorized role extraction
                                    roles_dict = extract_roles_from_text(line)
                                    if roles_dict['project_roles']:
                                        project['project_roles'] = roles_dict['project_roles'][:2]
                                    if roles_dict['handled_roles']:
                                        project['handled_roles'] = roles_dict['handled_roles'][:3]
                                    
                                    if project.get('name'):
                                        projects.append(project)
        
        logger.info(f"Text fallback extracted: {len(contacts)} contacts, {len(projects)} projects")
    
    except Exception as e:
        logger.warning(f"Text fallback extraction failed: {e}")
    
    return {'contacts': contacts, 'projects': projects}

# ============================================================================
# Main Parser
# ============================================================================

def parse_pdf(pdf_path: str) -> Dict:
    """
    Enhanced PDF parsing with improved categorized role extraction
    """
    
    logger.info(f"Parsing PDF: {pdf_path}")
    
    company_info = extract_company_info(pdf_path)
    
    # Extract all tables
    all_tables = []
    
    # Camelot Lattice
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
    
    # Camelot Stream
    try:
        logger.info("Extracting tables with Camelot stream...")
        
        stream_configs = [
            {},
            {'edge_tol': 50, 'row_tol': 10, 'column_tol': 5},
            {'edge_tol': 100, 'row_tol': 15, 'column_tol': 10},
            {'edge_tol': 200, 'row_tol': 20, 'column_tol': 15},
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
                is_dup = False
                for existing_df, _, _, _ in all_tables:
                    if existing_df.shape == table.df.shape:
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
    
    # pdfplumber
    try:
        pdfplumber_tables = extract_tables_with_pdfplumber(pdf_path)
        
        for df, page, method, confidence in pdfplumber_tables:
            is_dup = False
            for existing_df, _, _, _ in all_tables:
                if existing_df.shape == df.shape:
                    try:
                        if np.array_equal(existing_df.values, df.values):
                            is_dup = True
                            break
                    except:
                        pass
            
            if not is_dup:
                all_tables.append((df, page, method, confidence))
        
        logger.info(f"Added {len(pdfplumber_tables)} pdfplumber tables")
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed: {e}")
    
    logger.info(f"Total tables to process: {len(all_tables)}")
    
    # Process tables
    contacts = []
    projects = []
    tenders = []
    quality_scores = []
    
    for df, page, method, accuracy in all_tables:
        table_type, confidence = detect_table_type(df)
        
        logger.info(
            f"Page {page}, {method}: {table_type} "
            f"(confidence: {confidence:.2f}, accuracy: {accuracy:.0f}%, shape: {df.shape})"
        )
        
        if table_type == 'unknown' or confidence < 0.3:
            continue
        
        quality_scores.append(confidence)
        
        if table_type == 'contact':
            new_contacts = extract_contacts_from_table(df)
            contacts.extend(new_contacts)
            
        elif table_type == 'project':
            new_projects = extract_projects_from_table(df)
            projects.extend(new_projects)
            
        elif table_type == 'tender':
            new_tenders = extract_tenders_from_table(df)
            tenders.extend(new_tenders)
    
    # Text fallback
    if len(contacts) < 1 or len(projects) < 2:
        logger.info("Low extraction results - trying text fallback...")
        fallback = extract_from_text_fallback(pdf_path)
        
        if fallback['contacts']:
            logger.info(f"Text fallback found {len(fallback['contacts'])} additional contacts")
            contacts.extend(fallback['contacts'])
        
        if fallback['projects']:
            logger.info(f"Text fallback found {len(fallback['projects'])} additional projects")
            projects.extend(fallback['projects'])
    
    # Deduplicate
    contacts = deduplicate_contacts(contacts)
    projects = deduplicate_projects(projects)
    
    # Quality metrics
    avg_confidence = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    methods_used = list(set([method for _, _, method, _ in all_tables]))
    if len(contacts) > 0 or len(projects) > 0:
        if not quality_scores:
            methods_used.append('text-fallback')
    
    logger.info(
        f"Final results: {len(contacts)} contacts, "
        f"{len(projects)} projects, {len(tenders)} tenders"
    )
    
    # Log role extraction success
    contacts_with_project = sum(1 for c in contacts if c.get('project_roles'))
    contacts_with_handled = sum(1 for c in contacts if c.get('handled_roles'))
    projects_with_project = sum(1 for p in projects if p.get('project_roles'))
    projects_with_handled = sum(1 for p in projects if p.get('handled_roles'))
    
    logger.info(
        f"Roles extracted - Contacts (project: {contacts_with_project}, handled: {contacts_with_handled}), "
        f"Projects (project: {projects_with_project}, handled: {projects_with_handled})"
    )
    
    return {
        'company_info': company_info,
        'contacts': contacts,
        'projects': projects,
        'tenders': tenders,
        'quality': {
            'avg_confidence': round(avg_confidence, 2),
            'tables_processed': len(quality_scores),
            'extraction_methods': methods_used
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
            
            text = pdf.pages[0].extract_text() or ""
            lines = text.split('\n')[:40]
            
            for line in lines:
                line = line.strip()
                
                if 'cvr' in line.lower() or 'org nr' in line.lower() or 'org. nr' in line.lower():
                    match = re.search(r'\b(\d{8})\b', line)
                    if match:
                        info['cvr'] = match.group(1)
                
                if 'id nr' in line.lower():
                    match = re.search(r'\b(\d+)\b', line)
                    if match:
                        info['id_nr'] = match.group(1)
                
                if 'email' not in info:
                    emails = extract_emails(line)
                    if emails:
                        info['email'] = emails[0]
                
                if 'http' in line.lower():
                    match = re.search(r'(https?://[^\s]+)', line)
                    if match:
                        info['website'] = match.group(1)
                
                if 'phone' not in info:
                    if any(word in line.lower() for word in ['telefon', 'phone', 'tlf', 'mobil']):
                        if 'cvr' not in line.lower() and 'org nr' not in line.lower():
                            phones = extract_phones(line)
                            if phones:
                                info['phone'] = phones[0]
                
                if 'name' not in info:
                    if any(suffix in line for suffix in [' A/S', ' ApS', ' A.S', ' IVS', ' I/S']):
                        if len(line) < 80 and not line.isupper():
                            info['name'] = line
    
    except Exception as e:
        logger.warning(f"Failed to extract company info: {e}")
    
    return info
