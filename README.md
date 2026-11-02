# 🏫 Campus Access AI

### AI-Powered Smart Campus Access and Monitoring System

A smart campus security and access management prototype combining QR-based identity verification, category-based access control, CCTV person detection, and user status management.

## 🎯 Problem

Managing access to a large campus can be difficult when students, employees, faculty, workers, and visitors have different access permissions.

Traditional ID-card systems may not provide real-time verification or easily handle changes such as graduation, expiry, or account deactivation.

## 💡 Our Solution

Campus Access AI provides a centralized system where campus users can be registered, assigned a unique QR code, verified during entry, and granted or denied access based on their category and current status.

The system also provides a CCTV monitoring prototype using AI-based person detection.

## ✨ Key Features

- Student and employee registration
- Unique QR code generation
- QR-based identity verification
- Category-based access permissions
- Active / inactive / expired status
- AI-based CCTV person detection
- Person counting
- Admin monitoring dashboard
- User deactivation

## 🔄 Main Workflow

```text
User Registration
       ↓
Unique QR Code Generated
       ↓
QR Code Scanned
       ↓
Identity Retrieved
       ↓
Category & Status Verified
       ↓
ACTIVE → ACCESS GRANTED
INACTIVE / EXPIRED → ACCESS DENIED
```

## 📹 CCTV Monitoring

The system is planned to use computer vision to detect people from a CCTV feed or sample video.

## 🧑‍💻 Technology Stack

- Python
- SQLite
- OpenCV
- YOLO
- QR Code
- Streamlit
- REST API

## 🗃️ Identity Categories

- Student
- Faculty
- Employee
- Food Outlet Worker
- Visitor

## 🔐 Access Status

- ACTIVE
- INACTIVE
- EXPIRED

## 🚀 Prototype Goal

Register a student → Generate QR → Scan QR → Verify identity → Grant access → Change status → Scan again → Deny access.

The CCTV module demonstrates AI-based person detection alongside the access management system.

## ⚠️ Hackathon Prototype

This is a rapid hackathon prototype. Sample data and sample video footage may be used where live campus infrastructure is unavailable.

## 👥 Team

Hackathon Team – GDG DevCampus 2026

## 📌 Project Status

🚧 Currently under development
