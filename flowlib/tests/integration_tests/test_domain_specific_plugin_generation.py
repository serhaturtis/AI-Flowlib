#!/usr/bin/env python3
"""Integration test for domain-specific knowledge plugin generation.

This test compares different domain strategies to validate that domain-specific
extraction produces more targeted and higher-quality results for specialized content.
"""

import asyncio
import json
import logging
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

from flowlib.resources.decorators.decorators import llm_config, model_config
from flowlib.resources.models.config_resource import LLMConfigResource
from flowlib.resources.models.model_resource import ModelResource
from flowlib.knowledge.plugin_generation.flow import PluginGenerationFlow
from flowlib.knowledge.plugin_generation.models import (
    PluginGenerationRequest, DomainStrategy, DomainStrategyConfig
)


# Domain-specific test configurations (clean separation - model-specific settings only)
@model_config(
    name="knowledge-extraction", 
    provider_type="llamacpp",
    config={
        "path": "/home/swr/tools/llm_models/phi-4-Q6_K.gguf",
        "model_type": "phi4",
        "n_ctx": 8192,  # Model-specific context window
        "use_gpu": True,
        "n_gpu_layers": -1,  # Model-specific GPU usage
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1
        # n_threads, n_batch, verbose are now provider-level settings
    }
)
class DomainTestKnowledgeExtractionModel(ModelResource):
    """Domain test model configuration for knowledge extraction."""
    pass


@llm_config("default-llm")
class DomainTestLLMProvider(LLMConfigResource):
    """Domain test LLM provider configuration (clean separation - provider-level settings only)."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="llamacpp",
            # Provider-level settings only (n_ctx, use_gpu, n_gpu_layers are model-specific)
            **data
        )


class TestDomainSpecificPluginGeneration:
    """Test domain-specific plugin generation strategies."""
    
    @pytest.fixture
    def software_engineering_documents(self):
        """Create documents specifically for software engineering domain."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # API Documentation
        (temp_dir / "api_documentation.txt").write_text("""
REST API Design Documentation

Authentication Service API:
The UserAuthenticationService class implements JWT-based authentication. It exposes several endpoints:

POST /api/auth/login
- Endpoint: LoginEndpoint extends BaseEndpoint
- Method: authenticate(credentials: UserCredentials) -> AuthToken
- Dependencies: UserRepository, TokenGenerator, PasswordValidator
- Returns: JWT token with user claims

GET /api/auth/verify
- Endpoint: TokenVerificationEndpoint implements AuthEndpoint
- Method: verifyToken(token: string) -> UserInfo
- Dependencies: JWTValidator, UserService
- Throws: InvalidTokenException, ExpiredTokenException

Database Layer:
UserRepository class inherits from BaseRepository<User>
- Method: findByEmail(email: string) -> User | null
- Method: save(user: User) -> User
- Dependencies: DatabaseConnection, UserMapper
- Database: PostgreSQL users table

Architecture:
- AuthController uses AuthenticationService
- AuthenticationService depends_on UserRepository
- UserRepository connects_to PostgreSQL database
- JWT tokens are stored_in Redis cache
- Frontend calls API endpoints via HTTPS
""")

        # Technical Architecture
        (temp_dir / "microservices_architecture.txt").write_text("""
Microservices System Architecture

Service Definitions:

UserService:
- Class: UserMicroservice extends BaseMicroservice
- Language: Java Spring Boot
- Database: PostgreSQL user_db
- Endpoints: /users/*, /profiles/*
- Dependencies: AuthService, NotificationService

OrderService:
- Class: OrderProcessingService implements OrderHandler
- Language: Node.js Express
- Database: MongoDB orders_collection
- Dependencies: UserService, PaymentService, InventoryService
- Message Queue: RabbitMQ order_events

PaymentService:
- Class: PaymentProcessor extends PaymentGateway
- Language: Python FastAPI
- External APIs: Stripe, PayPal
- Security: PCI compliance, TokenVault

Infrastructure:
- Kubernetes cluster manages all services
- Docker containers package each service
- Nginx load balancer routes requests
- Redis caches session data
- ELK stack handles logging
- Prometheus monitors metrics

Service Communication:
- UserService communicates_with OrderService via REST
- OrderService publishes_to order_events queue
- PaymentService subscribes_to payment_requests
- All services register_with Consul service discovery
""")

        # Code Structure
        (temp_dir / "code_structure.txt").write_text("""
Application Code Structure

Core Classes:

public class UserController extends BaseController {
    @Autowired private UserService userService;
    @Autowired private ValidationService validator;
    
    @GetMapping("/users/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        // UserController calls UserService.findById
        // UserService depends_on UserRepository
        // UserRepository queries user_database
    }
}

interface UserService extends CrudService<User> {
    User findById(Long id);
    List<User> findByRole(UserRole role);
    // UserService interface implemented_by UserServiceImpl
}

@Service
public class UserServiceImpl implements UserService {
    @Autowired private UserRepository repository;
    @Autowired private CacheManager cacheManager;
    
    // UserServiceImpl uses UserRepository
    // CacheManager caches user data in Redis
}

Build Configuration:
- Maven pom.xml manages dependencies
- Spring Boot framework provides IoC container
- JUnit tests verify functionality
- Gradle builds production artifacts
- Jenkins CI/CD pipeline deploys to Kubernetes

Dependencies:
- spring-boot-starter-web provides REST capabilities
- spring-boot-starter-data-jpa handles database access
- junit-jupiter enables unit testing
- mockito supports test mocking
- logback handles application logging
""")

        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def generic_documents(self):
        """Create general documents for comparison."""
        temp_dir = Path(tempfile.mkdtemp())
        
        (temp_dir / "general_content.txt").write_text("""
Modern Business Processes

Customer Relationship Management:
Companies use CRM systems to track customer interactions and manage sales pipelines. Key components include lead generation, opportunity management, and customer support workflows.

Project Management:
Teams utilize project management methodologies like Agile and Waterfall to deliver successful outcomes. Important concepts include sprint planning, stakeholder communication, and risk assessment.

Data Analytics:
Organizations leverage data analytics to make informed decisions. This involves data collection, statistical analysis, visualization, and reporting to identify trends and opportunities.
""")

        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_software_engineering_vs_generic_strategy(
        self, 
        software_engineering_documents: Path, 
        output_dir: Path
    ):
        """Compare Software Engineering domain strategy vs Generic strategy."""
        
        # Skip if model file doesn't exist
        model_path = Path("/home/swr/tools/llm_models/phi-4-Q6_K.gguf")
        if not model_path.exists():
            pytest.skip("Phi-4 model not available")
        
        # Test Generic Strategy
        generic_request = PluginGenerationRequest(
            input_directory=str(software_engineering_documents),
            output_directory=str(output_dir / "generic"),
            plugin_name="generic_software_plugin",
            description="Software docs with generic extraction",
            domains=["software_engineering"],
            domain_strategy=DomainStrategy.GENERIC,
            use_vector_db=False,
            use_graph_db=False,
            max_files=3,
            chunk_size=400,
            chunk_overlap=80
        )
        
        # Test Software Engineering Strategy
        se_config = DomainStrategyConfig(
            strategy=DomainStrategy.SOFTWARE_ENGINEERING,
            custom_entity_types=["microservice", "container", "api_gateway"],
            extraction_focus=["code_dependencies", "service_architecture", "data_flow"]
        )
        
        se_request = PluginGenerationRequest(
            input_directory=str(software_engineering_documents),
            output_directory=str(output_dir / "software_engineering"),
            plugin_name="se_software_plugin", 
            description="Software docs with SE-specific extraction",
            domains=["software_engineering"],
            domain_strategy=DomainStrategy.SOFTWARE_ENGINEERING,
            domain_config=se_config,
            use_vector_db=False,
            use_graph_db=False,
            max_files=3,
            chunk_size=400,
            chunk_overlap=80
        )
        
        # Run both extractions
        flow = PluginGenerationFlow()
        
        generic_result = await flow.run_pipeline(generic_request)
        se_result = await flow.run_pipeline(se_request)
        
        # Validate both succeeded
        assert generic_result.success, f"Generic strategy failed: {generic_result.error_message}"
        assert se_result.success, f"SE strategy failed: {se_result.error_message}"
        
        # Load and compare results
        generic_data = self._load_plugin_data(Path(generic_result.plugin_path))
        se_data = self._load_plugin_data(Path(se_result.plugin_path))
        
        # Validate extraction differences
        self._validate_domain_differences(generic_data, se_data)
        
        return {
            "generic": generic_data,
            "software_engineering": se_data,
            "comparison": self._compare_extractions(generic_data, se_data)
        }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_software_engineering_entity_specificity(
        self, 
        software_engineering_documents: Path, 
        output_dir: Path
    ):
        """Test that SE strategy extracts software-specific entities."""
        
        model_path = Path("/home/swr/tools/llm_models/phi-4-Q6_K.gguf")
        if not model_path.exists():
            pytest.skip("Phi-4 model not available")
        
        se_request = PluginGenerationRequest(
            input_directory=str(software_engineering_documents),
            output_directory=str(output_dir),
            plugin_name="se_specificity_test",
            description="Test SE entity specificity",
            domains=["software_engineering"],
            domain_strategy=DomainStrategy.SOFTWARE_ENGINEERING,
            use_vector_db=False,
            use_graph_db=False,
            max_files=3,
            chunk_size=300,
            chunk_overlap=50
        )
        
        flow = PluginGenerationFlow()
        result = await flow.run_pipeline(se_request)
        
        assert result.success, f"SE extraction failed: {result.error_message}"
        
        # Load extracted data
        plugin_data = self._load_plugin_data(Path(result.plugin_path))
        entities = plugin_data["entities"]
        relationships = plugin_data["relationships"]
        
        # Check for software-specific entities
        entity_names = [e.get('name', '').lower() for e in entities]
        entity_types = [e.get('entity_type', '') for e in entities]
        
        # Expected software engineering entities
        expected_se_entities = [
            "api", "endpoint", "service", "database", "repository", 
            "controller", "microservice", "jwt", "rest"
        ]
        
        found_se_entities = []
        for expected in expected_se_entities:
            if any(expected in name for name in entity_names):
                found_se_entities.append(expected)
        
        assert len(found_se_entities) >= 3, f"Should find SE entities. Found: {found_se_entities}"
        
        # Check for software-specific relationships
        rel_types = [r.get('relationship_type', '') for r in relationships]
        expected_se_relationships = [
            "implements", "extends", "depends_on", "calls", "uses", "inherits"
        ]
        
        found_se_relationships = [rt for rt in rel_types if rt in expected_se_relationships]
        assert len(found_se_relationships) >= 2, f"Should find SE relationships. Found: {set(found_se_relationships)}"
        
        return plugin_data
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_domain_strategy_prompt_customization(
        self,
        software_engineering_documents: Path,
        output_dir: Path
    ):
        """Test that SE strategy uses different prompts and produces different results."""
        
        model_path = Path("/home/swr/tools/llm_models/phi-4-Q6_K.gguf")
        if not model_path.exists():
            pytest.skip("Phi-4 model not available")
        
        # Test with custom domain configuration
        custom_config = DomainStrategyConfig(
            strategy=DomainStrategy.SOFTWARE_ENGINEERING,
            custom_entity_types=["kubernetes_deployment", "docker_container", "ci_cd_pipeline"],
            custom_relationship_types=["deploys_to", "builds", "tests", "monitors"],
            extraction_focus=["infrastructure", "deployment", "monitoring"],
            validation_rules={"min_confidence": 0.7, "require_technical_context": True}
        )
        
        request = PluginGenerationRequest(
            input_directory=str(software_engineering_documents),
            output_directory=str(output_dir),
            plugin_name="custom_se_plugin",
            description="Custom SE extraction with specialized config",
            domains=["software_engineering", "devops"],
            domain_strategy=DomainStrategy.SOFTWARE_ENGINEERING,
            domain_config=custom_config,
            use_vector_db=False,
            use_graph_db=False,
            max_files=3,
            chunk_size=500,
            chunk_overlap=100
        )
        
        flow = PluginGenerationFlow()
        result = await flow.run_pipeline(request)
        
        assert result.success, f"Custom SE extraction failed: {result.error_message}"
        
        plugin_data = self._load_plugin_data(Path(result.plugin_path))
        
        # Validate high-quality extraction
        entities = plugin_data["entities"]
        high_conf_entities = [e for e in entities if e.get('confidence', 0) >= 0.7]
        
        # Should meet minimum confidence requirement
        conf_ratio = len(high_conf_entities) / len(entities) if entities else 0
        assert conf_ratio >= 0.6, f"Low confidence ratio with custom config: {conf_ratio:.2f}"
        
        return plugin_data
    
    def _load_plugin_data(self, plugin_path: Path) -> Dict[str, Any]:
        """Load plugin data for analysis."""
        data_dir = plugin_path / "data"
        
        with open(data_dir / "entities.json") as f:
            entities = json.load(f)
        
        with open(data_dir / "relationships.json") as f:
            relationships = json.load(f)
        
        with open(data_dir / "documents.json") as f:
            documents = json.load(f)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "documents": documents,
            "plugin_path": str(plugin_path)
        }
    
    def _validate_domain_differences(self, generic_data: Dict, se_data: Dict):
        """Validate that domain strategies produce different results."""
        
        generic_entities = generic_data["entities"]
        se_entities = se_data["entities"]
        
        # Should have extracted entities in both
        assert len(generic_entities) > 0, "Generic strategy should extract entities"
        assert len(se_entities) > 0, "SE strategy should extract entities"
        
        # Extract entity types for comparison
        generic_types = set(e.get('entity_type', 'unknown') for e in generic_entities)
        se_types = set(e.get('entity_type', 'unknown') for e in se_entities)
        
        # SE strategy should have more diverse entity types for technical content
        assert len(se_types) >= len(generic_types), "SE strategy should identify more specific entity types"
        
        # Check relationship types
        generic_rels = generic_data["relationships"]
        se_rels = se_data["relationships"]
        
        generic_rel_types = set(r.get('relationship_type', 'unknown') for r in generic_rels)
        se_rel_types = set(r.get('relationship_type', 'unknown') for r in se_rels)
        
        # SE strategy should identify more technical relationships
        technical_rel_types = {"implements", "extends", "depends_on", "calls", "uses", "inherits"}
        se_technical_rels = se_rel_types.intersection(technical_rel_types)
        generic_technical_rels = generic_rel_types.intersection(technical_rel_types)
        
        assert len(se_technical_rels) >= len(generic_technical_rels), "SE strategy should find more technical relationships"
    
    def _compare_extractions(self, generic_data: Dict, se_data: Dict) -> Dict[str, Any]:
        """Compare extraction results between strategies."""
        
        return {
            "entity_counts": {
                "generic": len(generic_data["entities"]),
                "software_engineering": len(se_data["entities"])
            },
            "relationship_counts": {
                "generic": len(generic_data["relationships"]),
                "software_engineering": len(se_data["relationships"])
            },
            "entity_types": {
                "generic": list(set(e.get('entity_type') for e in generic_data["entities"])),
                "software_engineering": list(set(e.get('entity_type') for e in se_data["entities"]))
            },
            "relationship_types": {
                "generic": list(set(r.get('relationship_type') for r in generic_data["relationships"])),
                "software_engineering": list(set(r.get('relationship_type') for r in se_data["relationships"]))
            }
        }


if __name__ == "__main__":
    """Run domain-specific tests manually."""
    
    logging.basicConfig(level=logging.INFO)
    
    async def run_domain_comparison_test():
        """Run the domain strategy comparison test manually."""
        
        test_instance = TestDomainSpecificPluginGeneration()
        
        import tempfile
        software_docs = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        
        try:
            # Create software engineering documents
            (software_docs / "test_api.txt").write_text("""
UserService API Documentation

class UserController {
    @Autowired UserService userService;
    
    @GetMapping("/users/{id}")
    ResponseEntity<User> getUser(@PathVariable Long id) {
        return userService.findById(id);
    }
}

interface UserService extends BaseService {
    User findById(Long id);
}

@Service
class UserServiceImpl implements UserService {
    @Autowired UserRepository repository;
}
""")
            
            result = await test_instance.test_software_engineering_vs_generic_strategy(
                software_docs, output
            )
            
            print("\n🎯 Domain Strategy Comparison Results:")
            print("=" * 60)
            
            comparison = result["comparison"]
            
            print(f"📊 Entity Extraction:")
            print(f"  Generic: {comparison['entity_counts']['generic']} entities")
            print(f"  Software Engineering: {comparison['entity_counts']['software_engineering']} entities")
            
            print(f"\n🔗 Relationship Extraction:")
            print(f"  Generic: {comparison['relationship_counts']['generic']} relationships") 
            print(f"  Software Engineering: {comparison['relationship_counts']['software_engineering']} relationships")
            
            print(f"\n🏷️  Entity Types:")
            print(f"  Generic: {comparison['entity_types']['generic']}")
            print(f"  Software Engineering: {comparison['entity_types']['software_engineering']}")
            
            print(f"\n🔗 Relationship Types:")
            print(f"  Generic: {comparison['relationship_types']['generic']}")
            print(f"  Software Engineering: {comparison['relationship_types']['software_engineering']}")
            
            print(f"\n✅ Domain-specific extraction test completed successfully!")
            
        finally:
            shutil.rmtree(software_docs, ignore_errors=True)
            shutil.rmtree(output, ignore_errors=True)
    
    asyncio.run(run_domain_comparison_test())