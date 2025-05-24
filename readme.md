# Spot The Ball AI Predictor

A Flask web app that uses AI to predict the correct coordinates in "spot the ball" competitions like BOTB.

## 🎯 Project Goals

- Provide accurate AI predictions for weekly BOTB competitions.
- Help users win or improve their own guesses using multiple AI models.
- Offer a simple, clean, mobile-friendly interface for users.

## 🧠 Features

### 👤 General Users
- View the latest competition image.
- Click "Predict" to get AI prediction overlay.
- Mobile responsive layout.

### 🚀 Pro Users (Subscription)
- View predictions from 5 different AI models.
- See average prediction.
- Access historical predictions and results.
- Invite-only sign-up with one-time use code.

### 🛠️ Admin Panel
- Upload new competition image.
- Trigger prediction on uploaded image.
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

## 💡 Notes

- All AI model predictions are visualized on top of the competition image.
- Coordinate normalization + scaling is handled internally.
- System uses multiple AI models trained on scraped historical BOTB data.

---

## 📬 Contact / Ideas

If you're interested in helping out, donating, or collaborating — stay tuned for updates!