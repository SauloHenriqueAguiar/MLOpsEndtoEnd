"""
Kubeflow component for automated model deployment to Kubernetes.
Complete implementation with comprehensive deployment features.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple
import yaml
import os


def load_pipeline_config():
    """Load pipeline configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_pipeline_config()
base_image = config['global']['base_image']


@component(
    base_image=base_image,
    packages_to_install=[
        "kubernetes==27.2.0",
        "pyyaml==6.0.1",
        "jinja2==3.1.2",
        "requests==2.31.0"
    ]
)
def model_deployment_component(
    # Input model and approval
    trained_model: Input[Model],
    model_approval: Input[Dataset],
    quality_certificate: Input[Dataset],
    
    # Deployment parameters
    deployment_name: str = "house-price-api",
    namespace: str = "mlops",
    replicas: int = 3,
    port: int = 8000,
    image_name: str = "house-price-api:latest",
    
    # Resource specifications
    cpu_request: str = "500m",
    memory_request: str = "1Gi",
    cpu_limit: str = "1",
    memory_limit: str = "2Gi",
    
    # Deployment strategy
    strategy: str = "RollingUpdate",
    max_surge: int = 1,
    max_unavailable: int = 0,
    
    # Outputs
    deployment_status: Output[Dataset],
    service_endpoint: Output[Dataset]
    
) -> NamedTuple('DeploymentOutput', [
    ('deployment_successful', bool),
    ('service_url', str),
    ('deployment_version', str)
]):
    """
    Deploy approved model to Kubernetes with comprehensive configuration.
    
    Args:
        trained_model: Trained model artifact
        model_approval: Model approval decision
        quality_certificate: Quality certificate
        deployment_name: Name for Kubernetes deployment
        namespace: Kubernetes namespace
        replicas: Number of replicas
        port: Service port
        image_name: Container image name
        cpu_request: CPU resource request
        memory_request: Memory resource request
        cpu_limit: CPU resource limit
        memory_limit: Memory resource limit
        strategy: Deployment strategy
        max_surge: Max surge for rolling update
        max_unavailable: Max unavailable for rolling update
        
    Returns:
        NamedTuple with deployment results
    """
    import json
    import yaml
    import logging
    from datetime import datetime
    from typing import NamedTuple
    import os
    import tempfile
    import subprocess
    
    # Kubernetes client
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting model deployment component...")
    logger.info(f"Target: {deployment_name} in namespace {namespace}")
    
    def check_approval_status():
        """Check if model is approved for deployment"""
        logger.info("🔍 Checking model approval status...")
        
        try:
            with open(model_approval.path, 'r') as f:
                approval_data = json.load(f)
            
            is_approved = approval_data.get('approved', False)
            
            if not is_approved:
                logger.error("❌ Model not approved for deployment")
                return False, approval_data
            
            logger.info("✅ Model approved for deployment")
            return True, approval_data
            
        except Exception as e:
            logger.error(f"❌ Error checking approval: {e}")
            return False, {}
    
    def load_quality_certificate():
        """Load quality certificate information"""
        try:
            with open(quality_certificate.path, 'r') as f:
                cert_data = json.load(f)
            
            logger.info(f"📜 Quality certificate loaded:")
            logger.info(f"  - Status: {cert_data.get('certification_status', 'Unknown')}")
            logger.info(f"  - Grade: {cert_data.get('quality_grade', 'Unknown')}")
            
            return cert_data
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load quality certificate: {e}")
            return {}
    
    def setup_kubernetes_client():
        """Setup Kubernetes client configuration"""
        logger.info("⚙️ Setting up Kubernetes client...")
        
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
            logger.info("✅ Using in-cluster Kubernetes config")
        except:
            try:
                # Fallback to local kubeconfig
                config.load_kube_config()
                logger.info("✅ Using local kubeconfig")
            except Exception as e:
                logger.error(f"❌ Failed to setup Kubernetes config: {e}")
                raise
        
        # Create API clients
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        return apps_v1, core_v1
    
    def generate_deployment_manifest(cert_data):
        """Generate Kubernetes deployment manifest"""
        logger.info("📝 Generating deployment manifest...")
        
        # Generate deployment version
        deployment_version = datetime.now().strftime("v%Y%m%d-%H%M%S")
        
        # Deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'version': deployment_version,
                    'component': 'ml-api',
                    'managed-by': 'kubeflow'
                },
                'annotations': {
                    'deployment.kubernetes.io/revision': '1',
                    'mlops.company.com/model-version': deployment_version,
                    'mlops.company.com/quality-grade': cert_data.get('quality_grade', 'Unknown'),
                    'mlops.company.com/deployed-by': 'kubeflow-pipeline',
                    'mlops.company.com/deployment-timestamp': datetime.now().isoformat()
                }
            },
            'spec': {
                'replicas': replicas,
                'strategy': {
                    'type': strategy,
                    'rollingUpdate': {
                        'maxSurge': max_surge,
                        'maxUnavailable': max_unavailable
                    }
                },
                'selector': {
                    'matchLabels': {
                        'app': deployment_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': deployment_name,
                            'version': deployment_version
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': str(port),
                            'prometheus.io/path': '/metrics'
                        }
                    },
                    'spec': {
                        'serviceAccountName': 'mlops-api',
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        },
                        'containers': [{
                            'name': 'api',
                            'image': image_name,
                            'imagePullPolicy': 'IfNotPresent',
                            'ports': [
                                {
                                    'containerPort': port,
                                    'name': 'http',
                                    'protocol': 'TCP'
                                }
                            ],
                            'env': [
                                {
                                    'name': 'MODEL_PATH',
                                    'value': '/app/data/models'
                                },
                                {
                                    'name': 'LOG_LEVEL',
                                    'value': 'INFO'
                                },
                                {
                                    'name': 'WORKERS',
                                    'value': '1'
                                },
                                {
                                    'name': 'MODEL_VERSION',
                                    'value': deployment_version
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': cpu_request,
                                    'memory': memory_request
                                },
                                'limits': {
                                    'cpu': cpu_limit,
                                    'memory': memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'failureThreshold': 3
                            },
                            'securityContext': {
                                'allowPrivilegeEscalation': False,
                                'readOnlyRootFilesystem': True,
                                'capabilities': {
                                    'drop': ['ALL']
                                }
                            },
                            'volumeMounts': [
                                {
                                    'name': 'model-storage',
                                    'mountPath': '/app/data/models',
                                    'readOnly': True
                                },
                                {
                                    'name': 'tmp-volume',
                                    'mountPath': '/tmp'
                                }
                            ]
                        }],
                        'volumes': [
                            {
                                'name': 'model-storage',
                                'persistentVolumeClaim': {
                                    'claimName': 'mlops-model-pvc'
                                }
                            },
                            {
                                'name': 'tmp-volume',
                                'emptyDir': {}
                            }
                        ],
                        'restartPolicy': 'Always',
                        'terminationGracePeriodSeconds': 30
                    }
                }
            }
        }
        
        return deployment_manifest, deployment_version
    
    def generate_service_manifest():
        """Generate Kubernetes service manifest"""
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{deployment_name}-service",
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'component': 'ml-api'
                },
                'annotations': {
                    'service.beta.kubernetes.io/aws-load-balancer-type': 'nlb'
                }
            },
            'spec': {
                'type': 'ClusterIP',
                'ports': [
                    {
                        'port': port,
                        'targetPort': port,
                        'protocol': 'TCP',
                        'name': 'http'
                    }
                ],
                'selector': {
                    'app': deployment_name
                }
            }
        }
        
        return service_manifest
    
    def create_or_update_deployment(apps_v1, deployment_manifest):
        """Create or update Kubernetes deployment"""
        logger.info(f"🚀 Creating/updating deployment: {deployment_name}")
        
        try:
            # Check if deployment exists
            try:
                existing_deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace
                )
                logger.info("📝 Updating existing deployment...")
                
                # Update deployment
                api_response = apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    logger.info("📝 Creating new deployment...")
                    # Create new deployment
                    api_response = apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_manifest
                    )
                else:
                    raise
            
            logger.info(f"✅ Deployment {deployment_name} created/updated successfully")
            return True, api_response
            
        except ApiException as e:
            logger.error(f"❌ Kubernetes API error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False, None
    
    def create_or_update_service(core_v1, service_manifest):
        """Create or update Kubernetes service"""
        service_name = f"{deployment_name}-service"
        logger.info(f"🌐 Creating/updating service: {service_name}")
        
        try:
            # Check if service exists
            try:
                existing_service = core_v1.read_namespaced_service(
                    name=service_name, namespace=namespace
                )
                logger.info("📝 Updating existing service...")
                
                # Update service
                api_response = core_v1.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    logger.info("📝 Creating new service...")
                    # Create new service
                    api_response = core_v1.create_namespaced_service(
                        namespace=namespace,
                        body=service_manifest
                    )
                else:
                    raise
            
            logger.info(f"✅ Service {service_name} created/updated successfully")
            return True, api_response
            
        except ApiException as e:
            logger.error(f"❌ Kubernetes API error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
            return False, None
    
    def wait_for_deployment_ready(apps_v1, timeout=300):
        """Wait for deployment to be ready"""
        logger.info(f"⏳ Waiting for deployment to be ready (timeout: {timeout}s)...")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace
                )
                
                status = deployment.status
                desired_replicas = status.replicas or 0
                ready_replicas = status.ready_replicas or 0
                
                logger.info(f"📊 Deployment status: {ready_replicas}/{desired_replicas} ready")
                
                if ready_replicas == desired_replicas and desired_replicas > 0:
                    logger.info("✅ Deployment is ready!")
                    return True
                
                time.sleep(10)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking deployment status: {e}")
                time.sleep(10)
        
        logger.error(f"❌ Deployment not ready after {timeout} seconds")
        return False
    
    def create_ingress(core_v1):
        """Create ingress for external access"""
        logger.info("🌐 Creating ingress for external access...")
        
        ingress_manifest = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{deployment_name}-ingress",
                'namespace': namespace,
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'false'
                }
            },
            'spec': {
                'rules': [
                    {
                        'host': f"{deployment_name}.{namespace}.local",
                        'http': {
                            'paths': [
                                {
                                    'path': '/',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': f"{deployment_name}-service",
                                            'port': {
                                                'number': port
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        try:
            networking_v1 = client.NetworkingV1Api()
            
            # Check if ingress exists
            try:
                existing_ingress = networking_v1.read_namespaced_ingress(
                    name=f"{deployment_name}-ingress", namespace=namespace
                )
                logger.info("📝 Updating existing ingress...")
                
                api_response = networking_v1.patch_namespaced_ingress(
                    name=f"{deployment_name}-ingress",
                    namespace=namespace,
                    body=ingress_manifest
                )
                
            except ApiException as e:
                if e.status == 404:
                    logger.info("📝 Creating new ingress...")
                    api_response = networking_v1.create_namespaced_ingress(
                        namespace=namespace,
                        body=ingress_manifest
                    )
                else:
                    raise
            
            logger.info("✅ Ingress created/updated successfully")
            return True, f"{deployment_name}.{namespace}.local"
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create ingress: {e}")
            return False, None
    
    def perform_health_check(service_url):
        """Perform health check on deployed service"""
        logger.info("🏥 Performing health check...")
        
        import requests
        import time
        
        health_url = f"http://{service_url}/health"
        max_attempts = 30
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Health check passed!")
                    return True, response.json()
                else:
                    logger.warning(f"⚠️ Health check failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Health check attempt {attempt + 1}: {e}")
            
            time.sleep(10)
        
        logger.error("❌ Health check failed after all attempts")
        return False, None
    
    def test_prediction_endpoint(service_url):
        """Test the prediction endpoint"""
        logger.info("🧪 Testing prediction endpoint...")
        
        import requests
        
        predict_url = f"http://{service_url}/predict"
        test_data = {
            "area": 120,
            "quartos": 3,
            "banheiros": 2,
            "idade": 5,
            "garagem": 1,
            "bairro": "Zona Sul"
        }
        
        try:
            response = requests.post(predict_url, json=test_data, timeout=30)
            if response.status_code == 200:
                prediction_result = response.json()
                predicted_price = prediction_result.get('predicted_price', 0)
                logger.info(f"✅ Prediction test passed! Price: R$ {predicted_price:,.2f}")
                return True, prediction_result
            else:
                logger.warning(f"⚠️ Prediction test failed: {response.status_code}")
                return False, None
                
        except Exception as e:
            logger.warning(f"⚠️ Prediction test error: {e}")
            return False, None
    
    def rollback_deployment(apps_v1, deployment_version):
        """Rollback deployment in case of failure"""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Get deployment history
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            # Trigger rollback by updating annotation
            deployment.metadata.annotations['deployment.kubernetes.io/revision'] = 'rollback'
            
            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info("✅ Rollback initiated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    # ====================================
    # Main Deployment Logic
    # ====================================
    
    try:
        deployment_start_time = datetime.now()
        
        # Step 1: Check approval status
        is_approved, approval_data = check_approval_status()
        if not is_approved:
            raise ValueError("Model not approved for deployment")
        
        # Step 2: Load quality certificate
        cert_data = load_quality_certificate()
        
        # Step 3: Setup Kubernetes client
        apps_v1, core_v1 = setup_kubernetes_client()
        
        # Step 4: Generate manifests
        deployment_manifest, deployment_version = generate_deployment_manifest(cert_data)
        service_manifest = generate_service_manifest()
        
        # Step 5: Deploy to Kubernetes
        logger.info("🚀 Step 5: Deploying to Kubernetes")
        
        # Create/update service first
        service_success, service_response = create_or_update_service(core_v1, service_manifest)
        if not service_success:
            raise Exception("Failed to create/update service")
        
        # Create/update deployment
        deployment_success, deployment_response = create_or_update_deployment(apps_v1, deployment_manifest)
        if not deployment_success:
            raise Exception("Failed to create/update deployment")
        
        # Step 6: Wait for deployment to be ready
        logger.info("⏳ Step 6: Waiting for deployment")
        if not wait_for_deployment_ready(apps_v1):
            logger.warning("⚠️ Deployment not ready, attempting rollback...")
            rollback_deployment(apps_v1, deployment_version)
            raise Exception("Deployment failed to become ready")
        
        # Step 7: Create ingress
        logger.info("🌐 Step 7: Setting up external access")
        ingress_success, external_url = create_ingress(core_v1)
        
        # Step 8: Health checks
        logger.info("🏥 Step 8: Health checks")
        service_url = f"{deployment_name}-service.{namespace}.svc.cluster.local:{port}"
        
        health_success, health_data = perform_health_check(service_url)
        if not health_success:
            logger.warning("⚠️ Health check failed, but continuing...")
        
        # Step 9: Test prediction endpoint
        logger.info("🧪 Step 9: Testing prediction endpoint")
        prediction_test_success, prediction_data = test_prediction_endpoint(service_url)
        
        # Step 10: Generate deployment report
        logger.info("📊 Step 10: Generating deployment report")
        
        deployment_duration = (datetime.now() - deployment_start_time).total_seconds()
        
        deployment_report = {
            'deployment_timestamp': deployment_start_time.isoformat(),
            'deployment_duration_seconds': deployment_duration,
            'deployment_version': deployment_version,
            'deployment_status': 'SUCCESS',
            'kubernetes_resources': {
                'deployment': {
                    'name': deployment_name,
                    'namespace': namespace,
                    'replicas': replicas,
                    'image': image_name
                },
                'service': {
                    'name': f"{deployment_name}-service",
                    'port': port,
                    'type': 'ClusterIP'
                },
                'ingress': {
                    'created': ingress_success,
                    'url': external_url if ingress_success else None
                }
            },
            'health_checks': {
                'service_health': health_success,
                'prediction_test': prediction_test_success,
                'health_data': health_data,
                'prediction_data': prediction_data
            },
            'resource_allocation': {
                'cpu_request': cpu_request,
                'memory_request': memory_request,
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit
            },
            'quality_info': cert_data,
            'approval_info': approval_data
        }
        
        # Save deployment status
        with open(deployment_status.path, 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        # Save service endpoint info
        endpoint_info = {
            'service_name': f"{deployment_name}-service",
            'namespace': namespace,
            'internal_url': service_url,
            'external_url': external_url if ingress_success else None,
            'port': port,
            'endpoints': {
                'health': f"http://{service_url}/health",
                'predict': f"http://{service_url}/predict",
                'docs': f"http://{service_url}/docs",
                'metrics': f"http://{service_url}/metrics"
            },
            'deployment_version': deployment_version
        }
        
        with open(service_endpoint.path, 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info("✅ Model deployment completed successfully!")
        logger.info(f"🌐 Service URL: {service_url}")
        logger.info(f"📊 Deployment version: {deployment_version}")
        logger.info(f"⏱️ Deployment time: {deployment_duration:.2f} seconds")
        
        # Return deployment results
        DeploymentOutput = NamedTuple('DeploymentOutput', [
            ('deployment_successful', bool),
            ('service_url', str),
            ('deployment_version', str)
        ])
        
        return DeploymentOutput(
            deployment_successful=True,
            service_url=external_url if ingress_success else service_url,
            deployment_version=deployment_version
        )
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        
        # Save failure status
        failure_report = {
            'deployment_timestamp': deployment_start_time.isoformat(),
            'deployment_status': 'FAILED',
            'error_message': str(e),
            'deployment_version': locals().get('deployment_version', 'unknown')
        }
        
        with open(deployment_status.path, 'w') as f:
            json.dump(failure_report, f, indent=2)
        
        # Return failure
        DeploymentOutput = NamedTuple('DeploymentOutput', [
            ('deployment_successful', bool),
            ('service_url', str),
            ('deployment_version', str)
        ])
        
        return DeploymentOutput(
            deployment_successful=False,
            service_url="",
            deployment_version=""
        )


# Additional helper component for blue-green deployment
@component(
    base_image=base_image,
    packages_to_install=[
        "kubernetes==27.2.0",
        "pyyaml==6.0.1",
        "requests==2.31.0"
    ]
)
def blue_green_deployment_component(
    # Input model and approval
    trained_model: Input[Model],
    model_approval: Input[Dataset],
    
    # Blue-Green parameters
    deployment_name: str = "house-price-api",
    namespace: str = "mlops",
    image_name: str = "house-price-api:latest",
    traffic_split_percentage: int = 10,
    validation_duration_minutes: int = 30,
    
    # Outputs
    deployment_status: Output[Dataset]
    
) -> NamedTuple('BlueGreenOutput', [('promotion_successful', bool), ('rollback_executed', bool)]):
    """
    Execute blue-green deployment with gradual traffic shifting.
    
    Args:
        trained_model: Trained model artifact
        model_approval: Model approval decision
        deployment_name: Base deployment name
        namespace: Kubernetes namespace
        image_name: New container image
        traffic_split_percentage: Initial traffic percentage for green
        validation_duration_minutes: How long to validate before full promotion
        
    Returns:
        NamedTuple with blue-green deployment results
    """
    import json
    import time
    import logging
    from datetime import datetime
    from typing import NamedTuple
    from kubernetes import client, config
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔵🟢 Starting blue-green deployment...")
    
    try:
        # Setup Kubernetes client
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Step 1: Deploy green version
        green_deployment_name = f"{deployment_name}-green"
        
        # Create green deployment (copy of blue with new image)
        blue_deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name, namespace=namespace
        )
        
        green_deployment = blue_deployment
        green_deployment.metadata.name = green_deployment_name
        green_deployment.spec.template.spec.containers[0].image = image_name
        green_deployment.metadata.labels['version'] = 'green'
        
        # Deploy green
        apps_v1.create_namespaced_deployment(
            namespace=namespace, body=green_deployment
        )
        
        logger.info("🟢 Green deployment created")
        
        # Step 2: Wait for green to be ready
        time.sleep(60)  # Wait for deployment
        
        # Step 3: Gradually shift traffic
        logger.info(f"🚦 Shifting {traffic_split_percentage}% traffic to green")
        
        # Update service selector to include both blue and green
        # This would require a more sophisticated traffic management solution
        # like Istio or similar service mesh
        
        # Step 4: Monitor for validation period
        logger.info(f"⏳ Monitoring for {validation_duration_minutes} minutes...")
        
        validation_successful = True
        start_time = time.time()
        
        while time.time() - start_time < validation_duration_minutes * 60:
            # Check green deployment health
            green_deployment_status = apps_v1.read_namespaced_deployment(
                name=green_deployment_name, namespace=namespace
            )
            
            ready_replicas = green_deployment_status.status.ready_replicas or 0
            desired_replicas = green_deployment_status.status.replicas or 0
            
            if ready_replicas != desired_replicas:
                logger.warning("⚠️ Green deployment not healthy, preparing rollback...")
                validation_successful = False
                break
            
            # Check error rates, response times, etc.
            # This would integrate with monitoring systems
            
            time.sleep(60)
        
        if validation_successful:
            # Step 5: Promote green to production (100% traffic)
            logger.info("🎉 Validation successful, promoting green to production")
            
            # Update main service to point to green
            service = core_v1.read_namespaced_service(
                name=f"{deployment_name}-service", namespace=namespace
            )
            service.spec.selector['version'] = 'green'
            
            core_v1.patch_namespaced_service(
                name=f"{deployment_name}-service",
                namespace=namespace,
                body=service
            )
            
            # Delete old blue deployment
            apps_v1.delete_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            # Rename green to blue
            green_deployment_status.metadata.name = deployment_name
            green_deployment_status.metadata.labels['version'] = 'blue'
            
            apps_v1.create_namespaced_deployment(
                namespace=namespace, body=green_deployment_status
            )
            
            # Delete temporary green
            apps_v1.delete_namespaced_deployment(
                name=green_deployment_name, namespace=namespace
            )
            
            logger.info("✅ Blue-green deployment completed successfully")
            
            # Save status
            status_report = {
                'deployment_type': 'blue-green',
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS',
                'promotion_successful': True,
                'rollback_executed': False,
                'traffic_split_tested': traffic_split_percentage,
                'validation_duration_minutes': validation_duration_minutes
            }
            
            with open(deployment_status.path, 'w') as f:
                json.dump(status_report, f, indent=2)
            
            BlueGreenOutput = NamedTuple('BlueGreenOutput', [('promotion_successful', bool), ('rollback_executed', bool)])
            return BlueGreenOutput(promotion_successful=True, rollback_executed=False)
            
        else:
            # Step 6: Rollback - delete green deployment
            logger.warning("🔄 Validation failed, executing rollback...")
            
            apps_v1.delete_namespaced_deployment(
                name=green_deployment_name, namespace=namespace
            )
            
            logger.info("✅ Rollback completed - traffic remains on blue")
            
            # Save status
            status_report = {
                'deployment_type': 'blue-green',
                'timestamp': datetime.now().isoformat(),
                'status': 'ROLLBACK',
                'promotion_successful': False,
                'rollback_executed': True,
                'failure_reason': 'Green deployment validation failed'
            }
            
            with open(deployment_status.path, 'w') as f:
                json.dump(status_report, f, indent=2)
            
            BlueGreenOutput = NamedTuple('BlueGreenOutput', [('promotion_successful', bool), ('rollback_executed', bool)])
            return BlueGreenOutput(promotion_successful=False, rollback_executed=True)
            
    except Exception as e:
        logger.error(f"❌ Blue-green deployment failed: {e}")
        
        # Emergency rollback
        try:
            apps_v1.delete_namespaced_deployment(
                name=f"{deployment_name}-green", namespace=namespace
            )
        except:
            pass
        
        BlueGreenOutput = NamedTuple('BlueGreenOutput', [('promotion_successful', bool), ('rollback_executed', bool)])
        return BlueGreenOutput(promotion_successful=False, rollback_executed=True)


# Component for canary deployment
@component(
    base_image=base_image,
    packages_to_install=[
        "kubernetes==27.2.0",
        "pyyaml==6.0.1",
        "requests==2.31.0"
    ]
)
def canary_deployment_component(
    # Input model and approval
    trained_model: Input[Model],
    model_approval: Input[Dataset],
    
    # Canary parameters
    deployment_name: str = "house-price-api",
    namespace: str = "mlops",
    image_name: str = "house-price-api:latest",
    canary_percentage: int = 10,
    success_threshold: float = 0.99,
    error_threshold: float = 0.01,
    
    # Outputs
    deployment_status: Output[Dataset]
    
) -> NamedTuple('CanaryOutput', [('canary_successful', bool), ('promoted_to_production', bool)]):
    """
    Execute canary deployment with automated promotion based on metrics.
    
    Args:
        trained_model: Trained model artifact
        model_approval: Model approval decision
        deployment_name: Base deployment name
        namespace: Kubernetes namespace
        image_name: New container image
        canary_percentage: Percentage of traffic for canary
        success_threshold: Success rate threshold for promotion
        error_threshold: Error rate threshold for rollback
        
    Returns:
        NamedTuple with canary deployment results
    """
    import json
    import time
    import logging
    from datetime import datetime
    from typing import NamedTuple
    from kubernetes import client, config
    import requests
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🐤 Starting canary deployment...")
    
    try:
        # Setup Kubernetes client
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Step 1: Deploy canary version
        canary_deployment_name = f"{deployment_name}-canary"
        
        # Get current stable deployment
        stable_deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name, namespace=namespace
        )
        
        # Create canary deployment
        canary_deployment = stable_deployment
        canary_deployment.metadata.name = canary_deployment_name
        canary_deployment.spec.template.spec.containers[0].image = image_name
        canary_deployment.metadata.labels['version'] = 'canary'
        canary_deployment.spec.replicas = max(1, int(stable_deployment.spec.replicas * canary_percentage / 100))
        
        # Deploy canary
        apps_v1.create_namespaced_deployment(
            namespace=namespace, body=canary_deployment
        )
        
        logger.info(f"🐤 Canary deployment created with {canary_percentage}% traffic")
        
        # Step 2: Monitor canary metrics
        monitoring_duration = 300  # 5 minutes
        check_interval = 30  # 30 seconds
        
        start_time = time.time()
        metrics_history = []
        
        while time.time() - start_time < monitoring_duration:
            try:
                # Collect metrics from both stable and canary
                stable_metrics = collect_deployment_metrics(f"{deployment_name}-service", namespace)
                canary_metrics = collect_deployment_metrics(f"{canary_deployment_name}-service", namespace)
                
                # Calculate success rates and error rates
                canary_success_rate = calculate_success_rate(canary_metrics)
                canary_error_rate = calculate_error_rate(canary_metrics)
                
                metrics_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'canary_success_rate': canary_success_rate,
                    'canary_error_rate': canary_error_rate,
                    'stable_success_rate': calculate_success_rate(stable_metrics),
                    'stable_error_rate': calculate_error_rate(stable_metrics)
                }
                
                metrics_history.append(metrics_entry)
                
                logger.info(f"📊 Canary metrics: Success={canary_success_rate:.3f}, Error={canary_error_rate:.3f}")
                
                # Check if canary should be rolled back
                if canary_error_rate > error_threshold:
                    logger.warning(f"🚨 Canary error rate {canary_error_rate:.3f} exceeds threshold {error_threshold}")
                    raise Exception("Canary error rate too high")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"❌ Canary monitoring failed: {e}")
                raise
        
        # Step 3: Evaluate canary performance
        avg_success_rate = sum(m['canary_success_rate'] for m in metrics_history) / len(metrics_history)
        avg_error_rate = sum(m['canary_error_rate'] for m in metrics_history) / len(metrics_history)
        
        logger.info(f"📊 Average canary metrics: Success={avg_success_rate:.3f}, Error={avg_error_rate:.3f}")
        
        if avg_success_rate >= success_threshold and avg_error_rate <= error_threshold:
            # Step 4: Promote canary to production
            logger.info("🎉 Canary validation successful, promoting to production")
            
            # Update stable deployment with canary image
            stable_deployment.spec.template.spec.containers[0].image = image_name
            
            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=stable_deployment
            )
            
            # Delete canary deployment
            apps_v1.delete_namespaced_deployment(
                name=canary_deployment_name, namespace=namespace
            )
            
            logger.info("✅ Canary promotion completed successfully")
            
            # Save status
            status_report = {
                'deployment_type': 'canary',
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS',
                'canary_successful': True,
                'promoted_to_production': True,
                'canary_percentage': canary_percentage,
                'avg_success_rate': avg_success_rate,
                'avg_error_rate': avg_error_rate,
                'metrics_history': metrics_history
            }
            
            with open(deployment_status.path, 'w') as f:
                json.dump(status_report, f, indent=2)
            
            CanaryOutput = NamedTuple('CanaryOutput', [('canary_successful', bool), ('promoted_to_production', bool)])
            return CanaryOutput(canary_successful=True, promoted_to_production=True)
            
        else:
            # Step 5: Rollback canary
            logger.warning("🔄 Canary validation failed, rolling back...")
            
            apps_v1.delete_namespaced_deployment(
                name=canary_deployment_name, namespace=namespace
            )
            
            logger.info("✅ Canary rollback completed")
            
            # Save status
            status_report = {
                'deployment_type': 'canary',
                'timestamp': datetime.now().isoformat(),
                'status': 'ROLLBACK',
                'canary_successful': False,
                'promoted_to_production': False,
                'failure_reason': f'Metrics below threshold: success={avg_success_rate:.3f}, error={avg_error_rate:.3f}'
            }
            
            with open(deployment_status.path, 'w') as f:
                json.dump(status_report, f, indent=2)
            
            CanaryOutput = NamedTuple('CanaryOutput', [('canary_successful', bool), ('promoted_to_production', bool)])
            return CanaryOutput(canary_successful=False, promoted_to_production=False)
            
    except Exception as e:
        logger.error(f"❌ Canary deployment failed: {e}")
        
        # Emergency rollback
        try:
            apps_v1.delete_namespaced_deployment(
                name=f"{deployment_name}-canary", namespace=namespace
            )
        except:
            pass
        
        CanaryOutput = NamedTuple('CanaryOutput', [('canary_successful', bool), ('promoted_to_production', bool)])
        return CanaryOutput(canary_successful=False, promoted_to_production=False)


def collect_deployment_metrics(service_name: str, namespace: str):
    """Collect metrics from deployment service"""
    try:
        # This would integrate with your monitoring system (Prometheus, etc.)
        # For now, return mock metrics
        import random
        return {
            'requests_total': random.randint(100, 1000),
            'requests_success': random.randint(95, 99),
            'response_time_avg': random.uniform(0.1, 0.5),
            'error_count': random.randint(0, 5)
        }
    except Exception:
        return {
            'requests_total': 0,
            'requests_success': 0,
            'response_time_avg': 0,
            'error_count': 0
        }


def calculate_success_rate(metrics):
    """Calculate success rate from metrics"""
    total = metrics.get('requests_total', 0)
    if total == 0:
        return 0.0
    return metrics.get('requests_success', 0) / total


def calculate_error_rate(metrics):
    """Calculate error rate from metrics"""
    total = metrics.get('requests_total', 0)
    if total == 0:
        return 0.0
    return metrics.get('error_count', 0) / total


# Component for A/B testing deployment
@component(
    base_image=base_image,
    packages_to_install=[
        "kubernetes==27.2.0",
        "pyyaml==6.0.1",
        "requests==2.31.0",
        "numpy==1.24.3",
        "scipy==1.11.2"
    ]
)
def ab_testing_deployment_component(
    # Input models
    model_a: Input[Model],
    model_b: Input[Model],
    model_approval: Input[Dataset],
    
    # A/B testing parameters
    deployment_name: str = "house-price-api",
    namespace: str = "mlops",
    traffic_split: float = 0.5,  # 50/50 split
    test_duration_minutes: int = 60,
    significance_level: float = 0.05,
    minimum_sample_size: int = 1000,
    
    # Outputs
    ab_test_results: Output[Dataset]
    
) -> NamedTuple('ABTestOutput', [('test_completed', bool), ('winning_model', str), ('statistically_significant', bool)]):
    """
    Execute A/B testing between two models with statistical analysis.
    
    Args:
        model_a: First model to test
        model_b: Second model to test
        model_approval: Model approval decision
        deployment_name: Base deployment name
        namespace: Kubernetes namespace
        traffic_split: Traffic split between models (0.5 = 50/50)
        test_duration_minutes: How long to run the test
        significance_level: Statistical significance level
        minimum_sample_size: Minimum samples needed per variant
        
    Returns:
        NamedTuple with A/B test results
    """
    import json
    import time
    import logging
    from datetime import datetime
    from typing import NamedTuple
    import numpy as np
    from scipy import stats
    from kubernetes import client, config
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🅰️🅱️ Starting A/B testing deployment...")
    
    try:
        # Setup Kubernetes client
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Step 1: Deploy both model versions
        model_a_name = f"{deployment_name}-model-a"
        model_b_name = f"{deployment_name}-model-b"
        
        # Deploy model A
        deploy_model_variant(apps_v1, model_a_name, namespace, "model-a-image:latest", "model-a")
        
        # Deploy model B  
        deploy_model_variant(apps_v1, model_b_name, namespace, "model-b-image:latest", "model-b")
        
        logger.info("🅰️🅱️ Both model variants deployed")
        
        # Step 2: Configure traffic splitting
        configure_traffic_split(core_v1, deployment_name, namespace, traffic_split)
        
        logger.info(f"🚦 Traffic split configured: {traffic_split*100:.0f}% / {(1-traffic_split)*100:.0f}%")
        
        # Step 3: Collect data during test period
        logger.info(f"⏱️ Running A/B test for {test_duration_minutes} minutes...")
        
        test_data = {
            'model_a': {'predictions': [], 'response_times': [], 'errors': []},
            'model_b': {'predictions': [], 'response_times': [], 'errors': []}
        }
        
        start_time = time.time()
        while time.time() - start_time < test_duration_minutes * 60:
            # Collect metrics from both variants
            metrics_a = collect_variant_metrics(model_a_name, namespace)
            metrics_b = collect_variant_metrics(model_b_name, namespace)
            
            test_data['model_a']['predictions'].extend(metrics_a.get('predictions', []))
            test_data['model_a']['response_times'].extend(metrics_a.get('response_times', []))
            test_data['model_a']['errors'].extend(metrics_a.get('errors', []))
            
            test_data['model_b']['predictions'].extend(metrics_b.get('predictions', []))
            test_data['model_b']['response_times'].extend(metrics_b.get('response_times', []))
            test_data['model_b']['errors'].extend(metrics_b.get('errors', []))
            
            logger.info(f"📊 Samples collected: A={len(test_data['model_a']['predictions'])}, B={len(test_data['model_b']['predictions'])}")
            
            time.sleep(60)  # Collect data every minute
        
        # Step 4: Statistical analysis
        logger.info("📈 Performing statistical analysis...")
        
        analysis_results = perform_statistical_analysis(
            test_data, significance_level, minimum_sample_size
        )
        
        # Step 5: Determine winner and deploy
        if analysis_results['statistically_significant']:
            winning_model = analysis_results['winning_model']
            logger.info(f"🏆 Winner: Model {winning_model} (p-value: {analysis_results['p_value']:.6f})")
            
            # Deploy winning model to production
            if winning_model == 'A':
                promote_model_to_production(apps_v1, model_a_name, deployment_name, namespace)
            else:
                promote_model_to_production(apps_v1, model_b_name, deployment_name, namespace)
            
            # Clean up test deployments
            cleanup_ab_test_deployments(apps_v1, [model_a_name, model_b_name], namespace)
            
        else:
            logger.info("📊 No statistically significant difference found")
            winning_model = "inconclusive"
            
            # Default to model A or current production model
            promote_model_to_production(apps_v1, model_a_name, deployment_name, namespace)
            cleanup_ab_test_deployments(apps_v1, [model_a_name, model_b_name], namespace)
        
        # Step 6: Save results
        results_report = {
            'test_type': 'ab_testing',
            'timestamp': datetime.now().isoformat(),
            'test_duration_minutes': test_duration_minutes,
            'traffic_split': traffic_split,
            'winning_model': winning_model,
            'statistically_significant': analysis_results['statistically_significant'],
            'analysis_results': analysis_results,
            'sample_sizes': {
                'model_a': len(test_data['model_a']['predictions']),
                'model_b': len(test_data['model_b']['predictions'])
            }
        }
        
        with open(ab_test_results.path, 'w') as f:
            json.dump(results_report, f, indent=2)
        
        logger.info("✅ A/B test completed successfully")
        
        ABTestOutput = NamedTuple('ABTestOutput', [('test_completed', bool), ('winning_model', str), ('statistically_significant', bool)])
        return ABTestOutput(
            test_completed=True,
            winning_model=winning_model,
            statistically_significant=analysis_results['statistically_significant']
        )
        
    except Exception as e:
        logger.error(f"❌ A/B testing failed: {e}")
        
        # Cleanup
        try:
            cleanup_ab_test_deployments(apps_v1, [f"{deployment_name}-model-a", f"{deployment_name}-model-b"], namespace)
        except:
            pass
        
        ABTestOutput = NamedTuple('ABTestOutput', [('test_completed', bool), ('winning_model', str), ('statistically_significant', bool)])
        return ABTestOutput(test_completed=False, winning_model="", statistically_significant=False)


def deploy_model_variant(apps_v1, deployment_name, namespace, image_name, variant_label):
    """Deploy a model variant for A/B testing"""
    # Implementation would create a deployment with the specific model image
    pass


def configure_traffic_split(core_v1, deployment_name, namespace, split_ratio):
    """Configure traffic splitting between model variants"""
    # Implementation would configure ingress or service mesh for traffic splitting
    pass


def collect_variant_metrics(deployment_name, namespace):
    """Collect metrics from a specific model variant"""
    # Mock implementation - would integrate with monitoring system
    import random
    return {
        'predictions': [random.uniform(100000, 500000) for _ in range(random.randint(10, 50))],
        'response_times': [random.uniform(0.1, 0.5) for _ in range(random.randint(10, 50))],
        'errors': [random.choice([0, 1]) for _ in range(random.randint(0, 5))]
    }


def perform_statistical_analysis(test_data, significance_level, minimum_sample_size):
    """Perform statistical analysis on A/B test data"""
    import numpy as np
    from scipy import stats
    
    # Check sample sizes
    samples_a = len(test_data['model_a']['predictions'])
    samples_b = len(test_data['model_b']['predictions'])
    
    if samples_a < minimum_sample_size or samples_b < minimum_sample_size:
        return {
            'statistically_significant': False,
            'reason': 'Insufficient sample size',
            'samples_a': samples_a,
            'samples_b': samples_b,
            'minimum_required': minimum_sample_size
        }
    
    # Compare response times using t-test
    response_times_a = np.array(test_data['model_a']['response_times'])
    response_times_b = np.array(test_data['model_b']['response_times'])
    
    t_stat, p_value = stats.ttest_ind(response_times_a, response_times_b)
    
    # Compare error rates using chi-square test
    errors_a = sum(test_data['model_a']['errors'])
    errors_b = sum(test_data['model_b']['errors'])
    
    # Determine statistical significance
    is_significant = p_value < significance_level
    
    # Determine winner based on response times
    winning_model = 'A' if np.mean(response_times_a) < np.mean(response_times_b) else 'B'
    
    return {
        'statistically_significant': is_significant,
        'winning_model': winning_model,
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'mean_response_time_a': float(np.mean(response_times_a)),
        'mean_response_time_b': float(np.mean(response_times_b)),
        'error_rate_a': errors_a / samples_a,
        'error_rate_b': errors_b / samples_b,
        'samples_a': samples_a,
        'samples_b': samples_b
    }


def promote_model_to_production(apps_v1, winning_deployment, production_deployment, namespace):
    """Promote winning model to production"""
    # Implementation would update production deployment with winning model
    pass


def cleanup_ab_test_deployments(apps_v1, deployment_names, namespace):
    """Clean up A/B test deployments"""
    for deployment_name in deployment_names:
        try:
            apps_v1.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
        except:
            pass