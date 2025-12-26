from django import template

register = template.Library()

@register.filter
def format_disease_name(value):
    """
    Format disease names from model output to readable format
    Example: 'Tomato___Bacterial_spot' -> 'Bacterial Spot'
    """
    if not value or '___' not in value:
        return value

    # Remove the 'Tomato___' prefix
    disease_part = value.split('___', 1)[1]

    # Replace underscores with spaces
    formatted = disease_part.replace('_', ' ')

    # Handle special cases
    formatted = formatted.replace('Two spotted spider mite', 'Two-spotted Spider Mite')

    return formatted

@register.filter
def disease_status(value):
    """
    Return 'healthy' or 'diseased' based on the result
    """
    return 'healthy' if value == 'Tomato___healthy' else 'diseased'