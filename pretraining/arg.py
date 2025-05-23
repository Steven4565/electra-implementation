import argparse
import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, get_type_hints

T = TypeVar('T')

class Field:
    """Base class for data field types."""
    
    @staticmethod
    def _is_optional(type_hint):
        """Check if a type hint is Optional[...]."""
        if hasattr(type_hint, '__origin__') and type_hint.__origin__ is Optional:
            return True
        return False

    @staticmethod
    def _get_optional_type(type_hint):
        """Get the type inside Optional[...]."""
        return type_hint.__args__[0]

    @staticmethod
    def convert(value, type_hint):
        """Convert string value to appropriate type."""
        if Field._is_optional(type_hint) and value is None:
            return None
        
        if Field._is_optional(type_hint):
            type_hint = Field._get_optional_type(type_hint)
            
        # Map custom field types to Python built-in types for conversion
        if type_hint is Str:
            actual_type = str
        elif type_hint is Int:
            actual_type = int
        elif type_hint is Float:
            actual_type = float
        elif type_hint is Bool:
            # Bool conversion is handled separately below
            pass 
        else:
            actual_type = type_hint

        if type_hint is bool or type_hint is Bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ('true', 't', 'yes', 'y', '1')
        return actual_type(value)


class Str(Field):
    """String data field type."""
    pass


class Int(Field):
    """Integer data field type."""
    pass


class Float(Field):
    """Float data field type."""
    pass


class Bool(Field):
    """Boolean data field type."""
    pass


def add_args(parser: argparse.ArgumentParser, args_class: Type[T]) -> None:
    """Add arguments from a dataclass to an argparse parser."""
    type_hints = get_type_hints(args_class)
    
    for field in dataclasses.fields(args_class):
        field_name = field.name
        field_type = type_hints[field_name]
        field_default = field.default
        
        # Skip ClassVar fields
        if hasattr(field_type, '__origin__') and field_type.__origin__ is ClassVar:
            continue
            
        # Handle Optional types
        is_optional = Field._is_optional(field_type)
        if is_optional:
            inner_type = Field._get_optional_type(field_type)
            field_type = inner_type
            
        # Convert field.default_factory() to value if it exists
        if field.default_factory is not dataclasses.MISSING:
            field_default = field.default_factory()
            
        # Add the argument to the parser
        help_str = f"Type: {field_type.__name__}"
        if field_default is not dataclasses.MISSING:
            help_str += f", Default: {field_default}"
            
        parser.add_argument(
            f"--{field_name}",
            type=str,
            default=None if field_default is dataclasses.MISSING else field_default,
            help=help_str,
            required=field_default is dataclasses.MISSING and not is_optional
        )


def parse_args(args_class: Type[T]) -> T:
    """Parse command-line arguments into a dataclass."""
    parser = argparse.ArgumentParser()
    add_args(parser, args_class)
    parsed_args = parser.parse_args()
    
    # Convert the parsed args to a dictionary
    args_dict = {}
    type_hints = get_type_hints(args_class)
    
    for field in dataclasses.fields(args_class):
        field_name = field.name
        field_type = type_hints[field_name]
        field_value = getattr(parsed_args, field_name)
        
        # Skip ClassVar fields
        if hasattr(field_type, '__origin__') and field_type.__origin__ is ClassVar:
            continue
            
        # Convert the value to the appropriate type
        if field_value is not None:
            args_dict[field_name] = Field.convert(field_value, field_type)
        
    # Create an instance of the dataclass
    return args_class(**args_dict) 