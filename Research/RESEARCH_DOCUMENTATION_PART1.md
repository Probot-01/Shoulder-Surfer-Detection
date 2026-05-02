# AI-Based Shoulder Surfing Prevention System
## Complete Research Documentation — PART 1 OF 5
### Sections Covered: Abstract · Introduction · Literature Background · System Architecture

---
**Author:** Solo College Project  
**Document Version:** 1.0  
**Date:** 2026  
**Classification:** Academic Research Documentation  

---

# SECTION 1: ABSTRACT

## 1.1 Overview

This document presents a complete technical and academic record of the design, development, training, integration, and evaluation of an AI-Based Shoulder Surfing Prevention System — a real-time, camera-driven privacy protection tool built entirely in Python and deployable on any standard laptop without requiring specialized hardware or a dedicated graphics processing unit.

## 1.2 Problem Statement

In an era where laptop computers are used in public, semi-public, and open-plan office environments as a matter of routine, the physical privacy of on-screen information has become a genuine and underappreciated security concern. Shoulder surfing — the act of observing another person's screen without their knowledge or consent — allows an attacker to silently capture passwords, read private messages, observe financial transactions, or view confidential business documents simply by looking over the victim's shoulder. Unlike network-based attacks, shoulder surfing requires no technical skill and leaves no digital trace. Despite its simplicity and effectiveness, no mainstream software solution exists that actively monitors for shoulder surfing in real time and responds automatically.

## 1.3 Proposed Solution

This project proposes and implements a seven-stage real-time computer vision pipeline that uses a standard webcam to continuously monitor the environment around the user's screen. The pipeline performs the following operations on every video frame: (1) captures a frame from the webcam, (2) uses YOLOv8n — a state-of-the-art real-time object detector — to locate all persons in the scene, (3) classifies each detected person as either the primary user or an observer based on their relative bounding box size, (4) extracts a padded face crop for each observer using MediaPipe FaceDetection, (5) classifies each observer's head pose as LOOKING or NOT LOOKING using a custom-trained MobileNetV2 binary classifier, (6) passes these per-observer results through a temporal decision engine that applies streak-based smoothing to eliminate momentary false detections, and (7) triggers a visual THREAT alert on the display when a sustained gaze is confirmed. The system achieves 55–65 frames per second with an observer present, with a confirmed-threat latency of approximately 150–200 milliseconds.

## 1.4 Key Results

The MobileNetV2 head pose classification model was trained on the BIWI Kinect Head Pose Database using a transfer-learning approach with the entire backbone frozen. The model achieved a validation accuracy of **91.99%** on a balanced binary classification task (LOOKING vs NOT LOOKING), trained in approximately 60–90 minutes on a Google Colab Tesla T4 GPU. At runtime, the complete detection-to-decision pipeline runs in real time on a CPU with no GPU required, achieving target frame rates suitable for practical deployment. The system correctly identifies observers who are actively looking at the screen while filtering out observers who are present but looking away, and maintains stable state across rapid scene changes via temporal smoothing.

## 1.5 Novelty

The core contribution of this project is the design of an **intent-based** rather than **presence-based** threat detection system. Prior work in screen privacy has either relied on physical filtering (privacy screens) or simple presence detection (detecting that a second person is nearby). This system goes further: it determines whether the second person is actively looking at the screen by analysing their head orientation, and only triggers a threat response when sustained directional gaze is confirmed. This makes the system dramatically more useful in practice — it does not trigger when a colleague sits nearby but faces their own screen, and it does not fail to trigger simply because the observer is close but clearly looking away. Additionally, the use of a probability gap filter (rather than raw argmax classification) and an asymmetric confidence threshold system represents a considered approach to balancing false positive and false negative rates in a safety-critical privacy application.

---

# SECTION 2: INTRODUCTION

## 2.1 Defining Shoulder Surfing

Shoulder surfing is a form of direct observation attack in which an adversary physically observes a victim's screen, keyboard, or other information display with the intent of capturing sensitive data. The term originates from the image of an attacker standing behind or beside a victim and peering over their shoulder. Unlike most cybersecurity threats, shoulder surfing requires no software tools, no network access, no malware, and no technical knowledge. The attacker uses only their eyes and physical proximity.

Shoulder surfing has been documented in academic security literature since at least the 1980s, when it was noted as a threat to ATM PIN entry. However, the proliferation of mobile devices, laptops, and open computing environments over the past two decades has dramatically expanded both the frequency and the potential severity of shoulder surfing attacks. Modern laptop screens display information of far greater sensitivity than an ATM keypad — entire email inboxes, banking dashboards, medical records, confidential business spreadsheets, private messages, and authentication credentials are routinely visible on screens in public and semi-public spaces.

### 2.1.1 What Constitutes a Shoulder Surfing Attack

A shoulder surfing attack can occur in any of the following forms:

**Direct observation:** The attacker stands or sits close enough to directly read the victim's screen. This is the most common form and requires no special equipment.

**Indirect observation via reflection:** The attacker uses a reflective surface — a window, a mirror, sunglasses, or even a polished table surface — to observe the screen from a different angle. This form is particularly difficult to detect because the attacker may appear to be looking away.

**Camera-assisted observation:** In high-value targets, an attacker may use a small camera or smartphone camera pointed at the victim's screen from a distance. This allows observation from further away and recording for later analysis.

**Collaborative observation:** Multiple people may work together, with one person distracting the victim while another reads the screen.

This project addresses the most common and widespread form: direct observation by a second person who is physically present and whose head is oriented toward the victim's screen.

## 2.2 Real-World Scenarios

### 2.2.1 Coffee Shops and Cafes

Coffee shops represent one of the highest-risk environments for shoulder surfing. Tables are placed close together, customers face in multiple directions, and the social norm of minding one's own business means victims rarely notice or respond to observers. A person seated at the adjacent table has a clear line of sight to any laptop screen. Remote workers and students frequently use coffee shops for extended work sessions, during which they may access email, banking applications, code repositories with sensitive API keys, or video calls with private participants.

### 2.2.2 Public Transport

Trains, buses, metros, and aeroplanes force people into close physical proximity, often for extended durations. Laptop users on commutes frequently process work emails or personal tasks. Seated arrangements on trains place passengers directly beside each other, and the angle of a laptop screen is frequently fully visible to adjacent passengers. Overhead passengers on trains and buses have a direct downward view of screens.

### 2.2.3 Airport Lounges and Waiting Areas

Business travellers routinely work on sensitive materials during layovers. Airport lounges have open seating, and waiting areas have rows of seats facing common directions. Charging stations create dense clusters of device users in close proximity. The combination of valuable targets (business travellers with corporate data) and high population density makes airports a particularly high-risk environment.

### 2.2.4 Open-Plan Offices

Modern office design trends have moved heavily toward open-plan layouts, hot-desking, and shared workspaces. While this promotes collaboration, it substantially increases exposure to shoulder surfing. In an open-plan office, a colleague walking past a workstation has a clear view of the screen. People in adjacent desks, people waiting for a meeting, or people standing at a nearby printer all have potential viewing angles to multiple screens simultaneously.

### 2.2.5 Libraries and Study Spaces

University libraries and public libraries have long tables where multiple students work in close proximity. A student working on research, coursework portals, or personal accounts has immediate neighbours on both sides. Study rooms with glass walls provide viewing angles from outside the room.

### 2.2.6 Healthcare Settings

Medical professionals who use computers at shared nursing stations or in patient consultation rooms may expose patient records to other patients, visitors, or unauthorized staff members. Healthcare data is among the most sensitive and legally protected categories of personal information; any unauthorized disclosure can have serious legal and ethical consequences.

## 2.3 Information at Risk

The range of information exposed during a shoulder surfing attack is broad and includes:

**Authentication credentials:** Passwords typed into login forms are visible character-by-character unless the input is obscured. Even an obscured field reveals the length of the password. Users who type slowly or who habitually look at the keyboard while typing are especially vulnerable.

**Banking and financial data:** Online banking dashboards, payment confirmations, account balances, and transaction histories are frequently accessed on laptops in public. A single observed account number and routing number is sufficient for significant financial fraud.

**Private messages and emails:** Email clients and messaging applications running on a laptop display sender names, subject lines, and message previews. A casual observer can read substantial personal or professional communication without the user's awareness.

**Medical and health information:** Patient portals, medical applications, and healthcare professional systems may display diagnoses, medications, and personal health records.

**Professional and confidential business documents:** Strategy documents, financial projections, source code, legal contracts, and HR records are routinely processed on laptops, particularly by remote workers.

**Two-factor authentication codes:** One-time passwords delivered via SMS or authenticator apps, if visible on an adjacent phone screen or the laptop screen itself, allow an observer who has separately obtained login credentials to complete an account takeover.

## 2.4 Why This Is a Growing Problem

Several converging trends have made shoulder surfing a more significant threat than it was even a decade ago:

**The remote work transition:** The widespread adoption of remote and hybrid work following 2020 means that workers who previously operated within the physical security controls of an office building now routinely work from home, coffee shops, libraries, and other uncontrolled environments. Corporate security policies designed for office buildings do not transfer to public spaces.

**Open office design philosophy:** Even within offices, the trend toward open-plan layouts, standing desks, and shared hot-desks has reduced the natural physical privacy barriers that traditional private offices or cubicle partitions provided.

**Laptop as primary device:** The shift from desktop computers (which face a wall or a fixed direction with a large back surface) to laptops (which can face any direction and have thin screens visible from both sides) has made screens more physically exposed.

**Increasing value of on-screen information:** The amount of sensitive information routinely accessed via a laptop has expanded dramatically. Where a 1990s office worker might access only work email, a modern worker accesses banking, medical records, personal and professional communication, and authentication systems all in the same session.

**Social norms around observation:** Despite its consequences, many people do not feel comfortable confronting an observer or even acknowledging that they have noticed one. This social friction means that shoulder surfing, when noticed, often goes unreported and unaddressed.

## 2.5 Existing Solutions and Their Inadequacy

### 2.5.1 Physical Privacy Screen Filters

Privacy screen filters are thin polarised film overlays that, when attached to a laptop screen, reduce the viewing angle such that the screen appears dark or black when viewed from more than approximately 30–45 degrees off-axis. They are available for most laptop screen sizes and are sold by manufacturers such as 3M, Kensington, and others.

**Why they are insufficient:**

*Static angle limitation:* Privacy screens only restrict the viewing angle in the horizontal plane (left-right). An observer seated directly behind the user at a slight elevation (a very common position on trains, in lecture halls, and in tiered seating) has a nearly direct on-axis view and is not affected by the privacy screen.

*Passive, non-adaptive:* A privacy screen applies the same restriction in all situations — whether the user is in a crowded airport or sitting alone in a private office. It provides no way to signal to the user when an observer is actually present, and provides no additional protection when none is needed.

*Purchase and attachment required:* The user must proactively purchase a screen filter, carry it with them, and attach it before working in a public space. Forgetting the filter, not owning one, or not anticipating the need leaves the user unprotected.

*Degraded user experience:* Privacy screens reduce overall screen brightness, change colour reproduction, and make the screen harder to read for the user as well. Many users find them uncomfortable for extended use.

*Cannot detect or respond:* A privacy screen cannot tell the user that an observer is present. It cannot escalate its protection when a threat is detected. It is a fixed physical filter with no intelligence.

### 2.5.2 Manual Screen Locking

Operating systems provide keyboard shortcuts (typically Win+L on Windows, Ctrl+Cmd+Q on macOS) to immediately lock the screen. Users can also set automatic lock timers.

**Why this is insufficient:**

*Requires user awareness:* The fundamental problem with shoulder surfing is that the user often does not know they are being observed. If the user had already noticed the observer, they would not need a system to detect the threat — they could respond directly. Manual screen locking solves the wrong problem.

*Reaction time:* Even when a user notices an observer, deciding to lock the screen, moving the hands to the keyboard, and executing the shortcut takes two to five seconds. A skilled or attentive observer can capture substantial information in that window.

*Workflow disruption:* Locking and unlocking the screen (re-entering a password) breaks workflow significantly. Users who find the process too disruptive will disable automatic locking or use very long timeout periods, leaving their screens exposed for extended periods.

*Automatic timers are blunt instruments:* Setting a 30-second or one-minute auto-lock means that every time the user pauses to think, their screen locks. This trains users to set longer timeouts, reducing the actual protection.

### 2.5.3 Absence of Real-Time AI Software Solutions

As of the time of this project, no mainstream software product exists that: (a) uses a standard webcam to monitor the environment around the user's screen, (b) detects the presence of observers in real time, (c) classifies whether those observers are actively looking at the screen, and (d) automatically activates screen protection when a confirmed threat is detected. Academic research papers have explored individual components of this problem (head pose estimation, gaze detection, person detection), but no integrated, deployable system addressing this specific use case has been published or released as a usable product.

## 2.6 The Proposed Solution

This project implements a fully automated, camera-based shoulder surfing detection system that requires no user action once started. The system runs continuously in the background, monitoring the webcam feed in real time. It uses the following approach:

1. **Continuous person detection** using YOLOv8n identifies all persons in the webcam's field of view at all times.

2. **Automatic user identification** assumes that the person whose bounding box is largest (and therefore closest to the camera, which is mounted on the laptop) is the primary user. This requires no calibration or registration.

3. **Per-observer head pose classification** extracts a face crop from each non-user person and classifies their head orientation as LOOKING (toward the screen) or NOT LOOKING using a trained MobileNetV2 model.

4. **Temporal threat decision** applies streak-based smoothing — requiring a sustained pattern of LOOKING frames across multiple consecutive video frames — before declaring a THREAT. This eliminates false positives from momentary glances.

5. **Automatic alert activation** changes the display state to THREAT when the decision engine reaches its threshold, providing immediate visual feedback in the detection window.

## 2.7 Project Scope and Constraints

This project was completed as a solo college project within approximately seven days using the following resource constraints:

- **Hardware:** Standard laptop with integrated webcam (640×480 resolution, 30 FPS). No external cameras, no depth sensors, no GPU at runtime.
- **Compute for training:** Google Colab free tier with Tesla T4 GPU (approximately 60–90 minutes of training time).
- **Dataset:** BIWI Kinect Head Pose Database (publicly available academic dataset, 24 subjects, ~15,000 images).
- **Software:** All open-source libraries (Python, PyTorch, OpenCV, MediaPipe, Ultralytics).
- **Budget:** Zero (all tools and datasets are free for academic use).

These constraints shaped every design decision in the project. The choice of MobileNetV2 over larger models, the use of YOLOv8n over YOLOv8l, the frame-skip optimisations, and the use of transfer learning rather than training from scratch are all direct consequences of operating within these constraints — and each decision is documented in detail in the subsequent sections of this document.

---

*PART 1 continues in SECTION 3 and SECTION 4 — see file RESEARCH_DOCUMENTATION_PART1B.md*
