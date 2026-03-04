"""User Service — Auth + Profiles + Follows — Section 05 / 40."""

from .routes import app
from .schemas import CreateUserRequest, CreateVendorRequest, UpdateProfileRequest

__all__ = ["app", "CreateUserRequest", "CreateVendorRequest", "UpdateProfileRequest"]
