# 🎯 WheresTheBall: AI-Powered Spot The Ball Predictor

WheresTheBall is a Flask web application that leverages artificial intelligence to predict the correct coordinates in "spot the ball" competitions, such as BOTB.

## 🚀 Project Status

The project is in its early stages, with the following components implemented:

- **Backend:** Flask framework set up with basic routes and configurations.
- **Frontend:** Basic HTML templates for the main page and about page.
- **Database:** SQLite database initialized with a sample structure.
- **AI Model:** Placeholder for AI model integration; actual model training and inference are yet to be implemented.

## 🧠 Features

### 👤 General Users

- View the latest competition image.
- Click "Predict" to get AI prediction overlay.
- Mobile-responsive layout.

### 🚀 Pro Users (Subscription)

- View predictions from multiple AI models.
- See average prediction.
- Access historical predictions and results.
- Invite-only sign-up with one-time use code.

### 🛠️ Admin Panel

- Upload new competition images.
- Trigger predictions on uploaded images.
- View, manage, and invite users.
- View prediction history.
- Invite users via one-time invite codes.

### 💳 Payments

- PayPal integration (IPN) for Pro subscription.
- Donation support.

### 🔐 Authentication & Security

- Email + password login.
- CAPTCHA on signup/login.
- Reset password via email.
- Auto logout from other sessions.
- Role-based access control.

### 🔍 SEO + Analytics

- SEO-friendly metadata and structure.
- Google Analytics integration.

## 🧰 Tech Stack

- **Backend:** Flask (Python)  
- **Frontend:** HTML, CSS, JS (vanilla or lightweight framework)  
- **DB:** SQLite (`database.db`)  
- **ML Models:** CNN-based coordinate regression (e.g., MobileNetV2 + ChannelAttention, heatmap regression, etc.)  
- **Payments:** PayPal (IPN)  
- **Scraping:** Python script to pull weekly images + data from BOTB  

## 🧪 In Progress / TODO

- User management UI  
- Admin dashboard UI polish  
- Pro section UI layout  
- Finalize model selection logic  
- Historical image filter system  
- Improve mobile UI/UX  
- Add Google reCAPTCHA  

---

## ⏱️ Estimated Timeline

Given the current state of the project, the following is an estimated timeline for completing the remaining features:

- **AI Model Integration:** 2–3 weeks  
- **Frontend Enhancements:** 1–2 weeks  
- **Admin Panel Development:** 2 weeks  
- **Payment Integration:** 1 week  
- **User Management & Authentication:** 1–2 weeks  
- **Testing & Deployment:** 1 week  

These estimates assume full-time development and relevant experience. Delays may occur depending on challenges or additional features.

---

## 📬 Contact / Ideas

If you're interested in helping out, donating, or collaborating — stay tuned for updates!
