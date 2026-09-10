import re
from typing import List, Dict, Any
from hiver_agent.intents.taxonomy import IntentDefinition, IntentTaxonomy
from hiver_agent.utils.logging import get_logger

logger = get_logger("intents.discovery")

AMAZON_INTENT_TAXONOMY_DEFAULT = IntentTaxonomy(
    brand="AmazonHelp",
    intents=[
        IntentDefinition(
            name="order_delivery_delay",
            description="Queries regarding package tracking, late delivery, missing shipment, or carrier updates.",
            inclusion_criteria=["Mentions tracking, delivery, package, missing order, late, carrier, shipped"],
            exclusion_criteria=["Demands money back or refund directly without tracking context"],
            examples=[
                "Where is my package? It was supposed to arrive yesterday.",
                "My order #123-456 says delivered but I haven't received it.",
                "Why is my shipment delayed by 3 days?"
            ],
            ambiguous_cases=["Package delivered broken -> classify as damaged_defective_item"]
        ),
        IntentDefinition(
            name="refund_return_status",
            description="Questions about return policy, return status, refund processing, or money reimbursement.",
            inclusion_criteria=["Mentions refund, return label, money back, return status, drop off"],
            exclusion_criteria=["Asking to cancel an order before shipping"],
            examples=[
                "When will I get my refund for order returned last week?",
                "How do I generate a return barcode for my item?",
                "I sent the item back 5 days ago, still no refund."
            ],
            ambiguous_cases=["Cancel order before dispatch -> classify as order_cancel_modify"]
        ),
        IntentDefinition(
            name="damaged_defective_item",
            description="Reports of receiving damaged, broken, defective, expired, or incorrect products.",
            inclusion_criteria=["Mentions damaged, broken, defective, missing parts, wrong item sent, expired"],
            exclusion_criteria=["General dissatisfaction without physical item defect"],
            examples=[
                "I opened the package and the glass bottle was shattered.",
                "You sent me size Small instead of Large.",
                "Item stopped working after 2 hours."
            ],
            ambiguous_cases=["Package box damaged but item fine -> damaged_defective_item if customer complains"]
        ),
        IntentDefinition(
            name="account_login_prime",
            description="Issues with Prime membership, subscription renewal, account locking, or login authorization.",
            inclusion_criteria=["Mentions Prime membership, subscription charge, account locked, OTP, sign in"],
            exclusion_criteria=["Payment declined for a specific physical item"],
            examples=[
                "I was charged $14.99 for Prime but I cancelled it last month.",
                "Cannot log into my Amazon account, OTP not received.",
                "How do I transfer Prime benefits to family?"
            ],
            ambiguous_cases=["Prime Video playback error -> classify as digital_services"]
        ),
        IntentDefinition(
            name="payment_billing_issue",
            description="Errors in payment processing, double charges, unauthorized credit card charges, or gift cards.",
            inclusion_criteria=["Mentions double charge, payment failed, card debited, gift card balance, promo code"],
            exclusion_criteria=["Prime subscription monthly fee -> account_login_prime"],
            examples=[
                "My credit card was charged twice for order #987.",
                "Gift card code says invalid when redeeming.",
                "Payment failed but money was deducted from bank account."
            ],
            ambiguous_cases=["Charged for Prime renewal -> account_login_prime"]
        ),
        IntentDefinition(
            name="digital_services",
            description="Technical glitches or issues with Prime Video, Music, Kindle, Fire TV, or Alexa.",
            inclusion_criteria=["Mentions Prime Video, Kindle ebook, Firestick, Alexa, streaming error, app crash"],
            exclusion_criteria=["Physical Kindle hardware broken -> damaged_defective_item"],
            examples=[
                "Prime Video giving error code 5004 on smart TV.",
                "Kindle book downloaded has missing chapters.",
                "Alexa won't connect to Wi-Fi after update."
            ],
            ambiguous_cases=["Kindle hardware screen cracked -> damaged_defective_item"]
        ),
        IntentDefinition(
            name="customer_service_complaint",
            description="Complaints regarding poor agent support, unhelpful phone reps, broken promises, or supervisory escalation.",
            inclusion_criteria=["Mentions rude representative, call disconnected, chat ended, supervisor, terrible service"],
            exclusion_criteria=["Specific shipping complaint without agent critique"],
            examples=[
                "Your chat agent disconnected me after waiting 30 minutes!",
                "I was promised a call back yesterday and no one called.",
                "Worst customer support experience ever."
            ],
            ambiguous_cases=["Delivery driver was rude -> order_delivery_delay / complaint"]
        ),
        IntentDefinition(
            name="general_inquiry_availability",
            description="Pre-purchase inquiries, item stock checks, price match, promotion rules, or feedback.",
            inclusion_criteria=["Mentions stock availability, price match, pre-order date, general question"],
            exclusion_criteria=["Questions about existing placed order"],
            examples=[
                "When will the PS5 be back in stock?",
                "Do you price match with Best Buy?",
                "Is this product covered under international warranty?"
            ],
            ambiguous_cases=["Inquiring when a placed order will ship -> order_delivery_delay"]
        ),
        IntentDefinition(
            name="out_of_scope_other",
            description="Messages that are greetings, spam, incomplete, uninterpretable, or outside customer support scope.",
            inclusion_criteria=["Generic hello, @mention with no text, emoji only, off-topic rant"],
            exclusion_criteria=["Any query expressing a specific problem"],
            examples=[
                "Hello @AmazonHelp",
                "DM sent",
                "???",
                "Cool!"
            ],
            ambiguous_cases=["DM sent with no detail -> out_of_scope_other"]
        )
    ]
)

def get_default_intent_taxonomy() -> IntentTaxonomy:
    return AMAZON_INTENT_TAXONOMY_DEFAULT
