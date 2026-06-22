from pydantic import BaseModel, Field
from typing import List, Optional

class ComponentSelection(BaseModel):
    name: str = Field(description="The name/model of the component.")
    price: float = Field(description="The price of the component.")
    specifications: str = Field(description="Key specs like socket type, wattage, size, etc.")

class PCConfiguration(BaseModel):
    cpu: Optional[ComponentSelection] = None
    motherboard: Optional[ComponentSelection] = None
    ram: Optional[ComponentSelection] = None
    gpu: Optional[ComponentSelection] = None
    storage: Optional[ComponentSelection] = None
    psu: Optional[ComponentSelection] = None
    case: Optional[ComponentSelection] = None
    total_price: float = Field(default=0.0, description="Sum total of all components.")
    is_compatible: bool = Field(default=False, description="Whether all selected components are mutually compatible.")
    compatibility_notes: str = Field(default="", description="Explanations regarding compatibility details (e.g., AM5 socket match, PSU wattage clearance).")


class UserRequirements(BaseModel):
    budget: Optional[float] = Field(None, description="The maximum budget specified by the user in USD or local currency.")
    primary_use: Optional[str] = Field(None, description="Primary use case, e.g., Gaming, Video Editing, Office Work, AI Training.")
    preferences: Optional[List[str]] = Field(default=[], description="Brand preferences (e.g., AMD, Intel, NVIDIA) or aesthetics (e.g., RGB, Mini-ITX).")
    is_ambiguous: bool = Field(..., description="Set to True if the input lacks critical information like budget or primary use.")
    is_conflicting: bool = Field(..., description="Set to True if the constraints are unrealistic (e.g., '$300 gaming PC to run Cyberpunk at 4K').")
    clarification_message: Optional[str] = Field(None, description="If ambiguous or conflicting, write a polite message to the user asking for specific missing details.")