"""Common exceptions for the application"""

class ServiceError(Exception):
    """Base exception for all service errors"""
    pass

class VoiceServiceError(ServiceError):
    """Exception for voice service errors"""
    pass

class SecurityError(ServiceError):
    """Exception for security-related errors"""
    pass

class AIServiceError(ServiceError):
    """Exception for AI service errors"""
    pass

class EmailServiceError(ServiceError):
    """Exception for email service errors"""
    pass

class WeatherServiceError(ServiceError):
    """Exception for weather service errors"""
    pass

class RAGServiceError(ServiceError):
    """Exception for RAG service errors"""
    pass

class ValidationError(ServiceError):
    """Exception for validation errors"""
    pass

class ConfigurationError(ServiceError):
    """Exception for configuration errors"""
    pass

class APIError(ServiceError):
    """Exception for API-related errors"""
    pass

class DatabaseError(ServiceError):
    """Exception for database-related errors"""
    pass

class AuthenticationError(SecurityError):
    """Exception for authentication errors"""
    pass

class AuthorizationError(SecurityError):
    """Exception for authorization errors"""
    pass
