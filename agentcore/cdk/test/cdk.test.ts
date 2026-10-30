import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { AgentCoreStack, addProjectPermissions } from '../lib/cdk-stack';

test('AgentCoreStack synthesizes with empty spec', () => {
  const app = new cdk.App();
  const stack = new AgentCoreStack(app, 'TestStack', {
    spec: {
      name: 'testproject',
      version: 1,
      managedBy: 'CDK' as const,
      runtimes: [],
      memories: [],
      credentials: [],
      evaluators: [],
      onlineEvalConfigs: [],
      configBundles: [],
      policyEngines: [],
      payments: [],
      agentCoreGateways: [],
      mcpRuntimeTools: [],
      unassignedTargets: [],
      datasets: [],
      knowledgeBases: [],
    },
  });
  const template = Template.fromStack(stack);
  template.hasOutput('StackNameOutput', {
    Description: 'Name of the CloudFormation Stack',
  });
});

// R5-07: the permissions the project adds to the Runtime role are not CLI output; a regenerated cdk/ would drop them
// unnoticed. They are checked on a role of their own, since the L3 construct needs build output to synthesize.
describe('addProjectPermissions', () => {
  const synth = (): Template => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'PermissionsStack');
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });
    addProjectPermissions(role, 'us-east-1', 'ACCOUNT');
    return Template.fromStack(stack);
  };

  test('the place tools may search', () => {
    synth().hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['geo-places:SearchNearby', 'geo-places:SearchText'],
            Effect: 'Allow',
            Resource: '*',
          }),
        ]),
      },
    });
  });

  test('the model id may be read from its one parameter and nothing else under /trv2/', () => {
    const template = synth();
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'ssm:GetParameter',
            Effect: 'Allow',
            Resource: 'arn:aws:ssm:us-east-1:ACCOUNT:parameter/trv2/*/agent-model-id',
          }),
        ]),
      },
    });
    expect(JSON.stringify(template.toJSON())).not.toContain('parameter/trv2/*"');
  });
});
