"""
Sample Data Generator for Project Sentinel
Creates realistic UFDR sample data for testing the preprocessing and chunking pipeline
"""

import json
import csv
import xml.etree.ElementTree as ET
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

class UFDRSampleDataGenerator:
    """Generates realistic UFDR sample data for testing"""
    
    def __init__(self):
        self.sample_names = [
            "Alex Rivera", "Sarah Mitchell", "Mike Rodriguez", "Chen Wei", 
            "Maria Santos", "James Wilson", "Priya Patel", "David Kim",
            "Anna Kowalski", "Carlos Mendez", "Fatima Al-Rashid", "Viktor Petrov"
        ]
        
        self.sample_emails = [
            "alex.r.crypto@tutanota.com", "sarah.m@protonmail.com", 
            "mike.rodriguez@secureemail.net", "chen.w@tempmail.org",
            "maria.santos@guerrillamail.com", "james.w@yopmail.com",
            "priya.p@mailinator.com", "david.k@10minutemail.com"
        ]
        
        self.crypto_addresses = {
            "bitcoin": [
                "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy", 
                "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
                "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
            ],
            "ethereum": [
                "0x742d35Cc6Bf8f8f54a8B4E1d5B1b8b4E4E4E4E4E",
                "0x8ba1f109551bD432803012645Hac136c22C6b9e5",
                "0xE92d1A43df510F82C66382592a047d288f85226f",
                "0x4E5B2e1dc63F6b91cb6Cd759936495434C7e972F"
            ]
        }
        
        self.suspicious_keywords = [
            "package", "delivery", "shipment", "transfer", "meeting",
            "location", "coordinates", "warehouse", "building", "operation",
            "payment", "crypto", "wallet", "bitcoin", "ethereum", "cash",
            "contact", "boss", "supplier", "overseas", "international"
        ]
        
        self.apps = ["WhatsApp", "Signal", "Telegram", "SMS", "iMessage"]
        
    def generate_phone_number(self, country_code: str = "+1") -> str:
        """Generate realistic phone number"""
        if country_code == "+1":  # US/Canada
            area_code = random.choice(["555", "202", "312", "415", "617", "212"])
            return f"+1-{area_code}-{random.randint(1000, 9999)}"
        elif country_code == "+44":  # UK
            return f"+44-7{random.randint(100, 999)}-{random.randint(100000, 999999)}"
        elif country_code == "+34":  # Spain
            return f"+34-{random.randint(600, 799)}-{random.randint(100, 999)}-{random.randint(100, 999)}"
        elif country_code == "+86":  # China
            return f"+86-{random.randint(130, 199)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        else:
            return f"{country_code}-{random.randint(100000000, 999999999)}"
    
    def generate_suspicious_message(self) -> str:
        """Generate suspicious message content"""
        templates = [
            "The {item} will arrive {time} at {location}. Be ready.",
            "Payment confirmed. Send to {crypto_address}. Make sure no trace.",
            "Meeting scheduled for {time} at coordinates {coordinates}. Come alone.",
            "Shipment from {location} delayed. New ETA is {time}.",
            "Boss wants to see the {item} before we proceed with phase 2.",
            "Transfer {amount} to account {account}. Use the usual method.",
            "Contact {contact} about the overseas {item}. Very important.",
            "Change of plans. New {location} for the exchange. Same time.",
            "Customs cleared. Package value: {amount}. Ready for pickup.",
            "Security protocols updated. Use authentication phrase: '{phrase}'"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            "item": random.choice(["package", "shipment", "delivery", "goods", "materials"]),
            "time": self._random_time(),
            "location": self._random_location(),
            "crypto_address": random.choice(self.crypto_addresses["bitcoin"] + self.crypto_addresses["ethereum"]),
            "coordinates": f"{random.uniform(40.7, 42.4):.4f}° N, {random.uniform(-74.2, -71.0):.4f}° W",
            "amount": f"{random.randint(10, 500)}k",
            "account": f"{random.randint(100000000, 999999999)}",
            "contact": random.choice(self.sample_names),
            "phrase": random.choice(["Evening Rain", "Blue Sky", "Silent Night", "Golden Dawn"])
        }
        
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    def _random_time(self) -> str:
        """Generate random time reference"""
        return random.choice([
            "tonight at 11 PM", "tomorrow morning", "Friday afternoon",
            "next Tuesday", "this weekend", "in 2 hours", "Monday at 3 PM"
        ])
    
    def _random_location(self) -> str:
        """Generate random location"""
        return random.choice([
            "the usual location", "building 7", "warehouse district",
            "downtown", "the parking garage", "pier 15", "terminal 3",
            "the office building", "the industrial area"
        ])
    
    def generate_xml_case(self, case_id: str, num_messages: int = 20, num_calls: int = 15) -> str:
        """Generate complete XML UFDR case"""
        
        # Create root element
        root = ET.Element("ufdr_report")
        
        # Case information
        case_elem = ET.SubElement(root, "case")
        ET.SubElement(case_elem, "id").text = case_id
        ET.SubElement(case_elem, "investigating_officer").text = f"Detective {random.choice(self.sample_names)}"
        ET.SubElement(case_elem, "date_created").text = datetime.now().isoformat()
        ET.SubElement(case_elem, "case_title").text = f"Operation {random.choice(['Digital Trail', 'Silent Network', 'Dark Web', 'Crypto Hunt'])}"
        
        # Device information
        device_elem = ET.SubElement(root, "device")
        ET.SubElement(device_elem, "make").text = random.choice(["Samsung", "Apple", "Google", "OnePlus"])
        ET.SubElement(device_elem, "model").text = random.choice(["Galaxy S23", "iPhone 14 Pro", "Pixel 7", "OnePlus 11"])
        ET.SubElement(device_elem, "imei").text = str(random.randint(100000000000000, 999999999999999))
        ET.SubElement(device_elem, "phone_number").text = self.generate_phone_number()
        
        # Communications
        comm_elem = ET.SubElement(root, "communications")
        messages_elem = ET.SubElement(comm_elem, "messages")
        
        contacts = []
        for i in range(5):  # Generate 5 contacts
            contacts.append({
                "name": self.sample_names[i],
                "phone": self.generate_phone_number(random.choice(["+1", "+44", "+34", "+86"])),
                "email": self.sample_emails[i % len(self.sample_emails)]
            })
        
        # Generate messages
        for i in range(num_messages):
            msg_elem = ET.SubElement(messages_elem, "message")
            ET.SubElement(msg_elem, "id").text = f"msg_{i+1:03d}"
            
            # Random timestamp within last 10 days
            timestamp = datetime.now() - timedelta(days=random.randint(1, 10), 
                                                 hours=random.randint(0, 23),
                                                 minutes=random.randint(0, 59))
            ET.SubElement(msg_elem, "timestamp").text = timestamp.isoformat()
            
            direction = random.choice(["incoming", "outgoing"])
            contact = random.choice(contacts)
            
            if direction == "incoming":
                ET.SubElement(msg_elem, "sender").text = contact["phone"]
                ET.SubElement(msg_elem, "recipient").text = self.generate_phone_number()
            else:
                ET.SubElement(msg_elem, "sender").text = self.generate_phone_number()
                ET.SubElement(msg_elem, "recipient").text = contact["phone"]
            
            ET.SubElement(msg_elem, "direction").text = direction
            ET.SubElement(msg_elem, "content").text = self.generate_suspicious_message()
            ET.SubElement(msg_elem, "app").text = random.choice(self.apps)
            ET.SubElement(msg_elem, "status").text = random.choice(["read", "delivered", "sent"])
        
        # Call logs
        calls_elem = ET.SubElement(root, "calls")
        for i in range(num_calls):
            call_elem = ET.SubElement(calls_elem, "call")
            ET.SubElement(call_elem, "id").text = f"call_{i+1:03d}"
            
            timestamp = datetime.now() - timedelta(days=random.randint(1, 10),
                                                 hours=random.randint(0, 23),
                                                 minutes=random.randint(0, 59))
            ET.SubElement(call_elem, "timestamp").text = timestamp.isoformat()
            ET.SubElement(call_elem, "direction").text = random.choice(["incoming", "outgoing"])
            ET.SubElement(call_elem, "number").text = random.choice(contacts)["phone"]
            ET.SubElement(call_elem, "duration").text = str(random.randint(30, 600))
            ET.SubElement(call_elem, "status").text = random.choice(["answered", "missed", "declined"])
        
        # Contacts
        contacts_elem = ET.SubElement(root, "contacts")
        for contact in contacts:
            contact_elem = ET.SubElement(contacts_elem, "contact")
            ET.SubElement(contact_elem, "name").text = contact["name"]
            ET.SubElement(contact_elem, "phone_number").text = contact["phone"]
            ET.SubElement(contact_elem, "email").text = contact["email"]
            ET.SubElement(contact_elem, "notes").text = f"Contact for {random.choice(['operations', 'logistics', 'finance', 'security'])}"
        
        # Convert to string
        return ET.tostring(root, encoding='unicode')
    
    def generate_csv_messages(self, num_messages: int = 30) -> List[Dict]:
        """Generate CSV format messages"""
        messages = []
        
        contacts = [
            {"name": name, "phone": self.generate_phone_number()} 
            for name in self.sample_names[:6]
        ]
        
        for i in range(num_messages):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 15),
                                                 hours=random.randint(0, 23),
                                                 minutes=random.randint(0, 59))
            
            contact = random.choice(contacts)
            direction = random.choice(["incoming", "outgoing"])
            
            message = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "app": random.choice(self.apps),
                "sender": contact["phone"] if direction == "incoming" else "+1-555-0123",
                "recipient": "+1-555-0123" if direction == "incoming" else contact["phone"],
                "message_content": self.generate_suspicious_message(),
                "status": random.choice(["read", "delivered", "sent"]),
                "direction": direction
            }
            
            messages.append(message)
        
        return sorted(messages, key=lambda x: x["timestamp"])
    
    def generate_json_case(self, case_id: str) -> Dict[str, Any]:
        """Generate JSON format case data"""
        
        contacts = [
            {
                "name": name,
                "phone": self.generate_phone_number(),
                "email": email,
                "notes": f"Contact for {random.choice(['operations', 'logistics', 'finance'])}"
            }
            for name, email in zip(self.sample_names[:5], self.sample_emails[:5])
        ]
        
        messages = []
        for i in range(25):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 10))
            contact = random.choice(contacts)
            direction = random.choice(["incoming", "outgoing"])
            
            messages.append({
                "timestamp": timestamp.isoformat(),
                "app": random.choice(self.apps),
                "direction": direction,
                "from": contact["phone"] if direction == "incoming" else "+1-555-0123",
                "to": "+1-555-0123" if direction == "incoming" else contact["phone"],
                "content": self.generate_suspicious_message(),
                "status": random.choice(["read", "delivered", "sent"])
            })
        
        calls = []
        for i in range(15):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 10))
            calls.append({
                "timestamp": timestamp.isoformat(),
                "direction": random.choice(["incoming", "outgoing"]),
                "number": random.choice(contacts)["phone"],
                "duration": random.randint(30, 600),
                "status": random.choice(["answered", "missed", "declined"])
            })
        
        return {
            "case_id": case_id,
            "device": {
                "make": random.choice(["Samsung", "Apple", "Google"]),
                "model": random.choice(["Galaxy S23", "iPhone 14 Pro", "Pixel 7"]),
                "imei": str(random.randint(100000000000000, 999999999999999)),
                "phone_number": "+1-555-0123",
                "os": random.choice(["Android 13", "iOS 16", "Android 14"])
            },
            "extraction_info": {
                "date": datetime.now().isoformat(),
                "method": "Physical",
                "tools_used": ["Cellebrite UFED", "Oxygen Detective Suite"],
                "examiner": f"Detective {random.choice(self.sample_names)}"
            },
            "messages": sorted(messages, key=lambda x: x["timestamp"]),
            "calls": sorted(calls, key=lambda x: x["timestamp"]),
            "contacts": contacts,
            "financial_data": {
                "crypto_addresses": [
                    {
                        "address": random.choice(self.crypto_addresses["bitcoin"]),
                        "type": "bitcoin",
                        "transactions": [
                            {
                                "amount": f"{random.randint(10, 100)}000 USD equivalent",
                                "timestamp": datetime.now().isoformat(),
                                "direction": "outgoing"
                            }
                        ]
                    }
                ],
                "bank_accounts": [
                    {
                        "account_number": str(random.randint(100000000, 999999999)),
                        "bank_name": random.choice(["Bank of America", "Chase", "Wells Fargo"]),
                        "type": "checking"
                    }
                ]
            }
        }
    
    def save_sample_data(self, output_dir: str = None):
        """Save all sample data types"""
        if output_dir is None:
            output_dir = settings.sample_data_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate XML case
        xml_data = self.generate_xml_case("CASE-2024-003", 25, 18)
        with open(output_path / "sample_case_003.xml", "w", encoding="utf-8") as f:
            f.write(xml_data)
        
        # Generate CSV messages
        csv_messages = self.generate_csv_messages(35)
        with open(output_path / "messages_case_003.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_messages[0].keys())
            writer.writeheader()
            writer.writerows(csv_messages)
        
        # Generate JSON case
        json_data = self.generate_json_case("CASE-2024-004")
        with open(output_path / "case_004_data.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample data generated and saved to {output_path}")
        return output_path

if __name__ == "__main__":
    generator = UFDRSampleDataGenerator()
    generator.save_sample_data()