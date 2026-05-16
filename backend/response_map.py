"""
backend/response_map.py
-----------------------
Intent-to-response mapping for the AI Chatbot.

Workflow:
  1. Defines INTENT_RESPONSES — a dictionary mapping every intent label
     (matching the Bitext dataset's 27 intent classes exactly) to a
     natural-language response string.
  2. Exposes get_response() which looks up the intent and returns the
     appropriate reply, with a safe fallback for unknown intents.

This module is imported by backend/predictor.py after classification
to convert the predicted intent label into a user-facing message.

IMPORTANT: Intent keys must match the dataset labels exactly
(case-sensitive). If you add new intents, add them here too.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Intent → Response mapping.
# All 27 intents from the Bitext Customer Support Intent Dataset.
# ---------------------------------------------------------------------------
INTENT_RESPONSES: dict[str, str] = {
    "cancel_order": (
        "I can help you cancel your order. Please provide your order number "
        "and I'll process the cancellation right away."
    ),
    "change_order": (
        "I can assist with modifying your order. Please share your order "
        "number and the changes you'd like to make."
    ),
    "change_shipping_address": (
        "To update your shipping address, please provide your order number "
        "and the new delivery address."
    ),
    "check_cancellation_fee": (
        "Cancellation fees depend on the order status. Orders cancelled "
        "before dispatch are free. After dispatch, a fee may apply. "
        "Share your order number for exact details."
    ),
    "check_invoices": (
        "Your invoices are available in your account under 'Order History'. "
        "I can also resend a specific invoice — just provide the order number."
    ),
    "check_payment_methods": (
        "We accept credit/debit cards (Visa, Mastercard, Amex), PayPal, "
        "and bank transfers. All payments are secured with SSL encryption."
    ),
    "check_refund_policy": (
        "Our refund policy allows returns within 30 days of delivery. "
        "Items must be unused and in original packaging. "
        "Refunds are processed within 5-7 business days."
    ),
    "complaint": (
        "I'm sorry to hear you've had a poor experience. Your feedback is "
        "important to us. Could you describe the issue so I can escalate "
        "it to the right team?"
    ),
    "contact_customer_service": (
        "You can reach our customer service team via email at "
        "support@example.com or by phone at 1-800-000-0000, "
        "Monday to Friday, 9 AM – 6 PM."
    ),
    "contact_human_agent": (
        "I'll connect you with a human agent right away. "
        "Please hold — average wait time is under 2 minutes."
    ),
    "create_account": (
        "Creating an account is easy! Visit our website, click 'Sign Up', "
        "and follow the steps. It only takes about 2 minutes."
    ),
    "delete_account": (
        "To permanently delete your account, go to Account Settings → "
        "Privacy → Delete Account. Note: this action cannot be undone "
        "and all your data will be removed."
    ),
    "delivery_options": (
        "We offer Standard Delivery (5-7 days), Express Delivery (2-3 days), "
        "and Same-Day Delivery in select areas. "
        "Shipping costs vary by location and order size."
    ),
    "delivery_period": (
        "Standard delivery takes 5-7 business days. Express takes 2-3 days. "
        "You'll receive a tracking link once your order is dispatched."
    ),
    "edit_account": (
        "You can update your account details — name, email, password, or "
        "address — by going to Account Settings on our website or app."
    ),
    "get_invoice": (
        "I can send your invoice to your registered email address. "
        "Please confirm your order number and I'll dispatch it immediately."
    ),
    "get_refund": (
        "To process your refund, I'll need your order number and the reason "
        "for the return. Refunds are credited within 5-7 business days "
        "to your original payment method."
    ),
    "newsletter_subscription": (
        "You can manage your newsletter preferences in Account Settings → "
        "Notifications. To unsubscribe, click 'Unsubscribe' at the bottom "
        "of any newsletter email."
    ),
    "payment_issue": (
        "I'm sorry you're experiencing a payment issue. Common causes include "
        "incorrect card details or insufficient funds. Please double-check "
        "your details or try a different payment method."
    ),
    "place_order": (
        "To place an order, browse our catalogue, add items to your cart, "
        "and proceed to checkout. Need help finding a specific product?"
    ),
    "recover_password": (
        "To reset your password, click 'Forgot Password' on the login page "
        "and enter your email. You'll receive a reset link within a few minutes."
    ),
    "registration_problems": (
        "Having trouble registering? Make sure your email isn't already in "
        "use and that your password meets our requirements (8+ characters, "
        "one number, one uppercase). Still stuck? Contact support."
    ),
    "review": (
        "We'd love your feedback! You can leave a review on your order from "
        "the 'Order History' page. Your insights help us improve."
    ),
    "set_up_shipping_address": (
        "To add a shipping address, go to Account Settings → Addresses → "
        "Add New Address. You can save multiple addresses for convenience."
    ),
    "switch_account": (
        "To switch between accounts, log out of your current account and "
        "log in with the other account's credentials."
    ),
    "track_order": (
        "To track your order, please provide your order number and I'll "
        "give you a real-time status update. You can also track via the "
        "link sent to your email."
    ),
    "track_refund": (
        "Refunds typically take 5-7 business days to appear. "
        "Please share your order number and I'll check the current "
        "status of your refund."
    ),
}

# Shown when the model predicts an intent not in the map,
# or confidence is too low.
_FALLBACK_RESPONSE = (
    "I'm not sure I understood that correctly. Could you please rephrase "
    "your question? You can also contact our support team at "
    "support@example.com for immediate assistance."
)


def get_response(intent: Optional[str]) -> str:
    """
    Return the customer-facing response string for a given intent label.

    Args:
        intent (Optional[str]): Predicted intent label, e.g. 'cancel_order'.
                                Pass None to get the fallback response.

    Returns:
        str: The mapped response text, or the fallback if intent is unknown.
    """
    if not intent:
        return _FALLBACK_RESPONSE

    return INTENT_RESPONSES.get(intent, _FALLBACK_RESPONSE)
