Spot The Ball AI Predictor
A Flask web app that uses AI-powered models to predict the exact location of the hidden ball in popular "spot the ball" competitions like BOTB. No more guesswork — get data-driven predictions to improve your chances of winning.

🎯 Project Overview
"Spot the Ball" competitions are fun but notoriously tricky — typically based on gut feelings and guesswork. This project flips the script by leveraging AI models trained on historical competition images and results to generate precise ball location predictions.

The app serves both casual users and dedicated pros by offering a simple, mobile-friendly interface and a subscription tier with multiple AI predictions and analytics.

🧠 Features
👤 General Users
View the latest competition image.

Get a single AI prediction overlay with the click of a button.

Responsive, clean UI optimized for desktop and mobile.

🚀 Pro Subscribers
Unlock 5 different AI model predictions for each competition image.

View the average prediction for better confidence.

Access full history of past competition images, predictions, and actual results.

Invite-only registration via one-time use codes.

🛠️ Admin Panel
Upload new competition images easily.

Trigger AI prediction jobs on new images.

Manage users, generate invite codes, and review prediction history.

💳 Payments & Donations
Integrated PayPal subscription payments with IPN callbacks.

Support the project with donations for ongoing AI model improvements.

🔐 Security
Secure email/password login.

CAPTCHA on sign-up and login to reduce spam.

Password reset via email.

Session management with auto logout.

Role-based access controls (user, admin, pro subscriber).

📈 SEO & Analytics
SEO-friendly page metadata.

Google Analytics tracking.

🧰 Tech Stack
Backend: Python Flask

Frontend: HTML, CSS, vanilla JS, Bootstrap 5

Database: SQLite (database.db)

Machine Learning: CNN-based coordinate regression models (MobileNetV2, heatmap regression, ChannelAttention layers, etc.)

Payments: PayPal IPN for subscription management

Scraping: Python scripts to scrape weekly competition images and results from BOTB

🚧 Work In Progress
Enhanced user management and UI polish.

Pro dashboard with prediction comparisons and overlays.

Advanced historical filtering and search.

Improved mobile UI and UX.

Google reCAPTCHA integration.

💡 How It Works
The AI models are trained on thousands of historical competition images and official judge results.

Predictions are coordinate points representing where the AI estimates the ball is hidden.

Multiple models provide diverse perspectives, and averaging them improves accuracy.

The app overlays these predictions on the competition images for easy visualization.

📬 Get Involved
Interested in contributing, donating, or collaborating? Reach out or follow the repo for updates!