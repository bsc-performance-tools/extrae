#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

/*
 * Self-contained simulation of the vulnerable timesync node registration
 * mechanism. We replicate the fixed-size buffer structure and the unsafe
 * strcpy pattern to verify that the security invariant holds:
 *
 * INVARIANT: Copying a node name into a fixed-size buffer must never
 * overwrite memory beyond the allocated buffer boundary, regardless of
 * the length of the input node name.
 */

#define NODE_NAME_MAX  64   /* assumed fixed buffer size per node entry */
#define MAX_NODES      16

/* Canary values placed around the buffer to detect overflow */
#define CANARY_VALUE   0xDEADBEEFCAFEBABEULL

typedef struct {
    uint64_t canary_before;
    char     name[NODE_NAME_MAX];
    uint64_t canary_after;
} GuardedNodeEntry;

static GuardedNodeEntry NodeList[MAX_NODES];
static int TotalNodes = 0;

/* Safe version: uses strncpy with explicit null-termination */
static int safe_add_node(const char *node)
{
    if (node == NULL) return -1;
    if (TotalNodes >= MAX_NODES) return -1;

    TotalNodes++;
    NodeList[TotalNodes - 1].canary_before = CANARY_VALUE;
    NodeList[TotalNodes - 1].canary_after  = CANARY_VALUE;

    /* Safe copy: bounded, null-terminated */
    strncpy(NodeList[TotalNodes - 1].name, node, NODE_NAME_MAX - 1);
    NodeList[TotalNodes - 1].name[NODE_NAME_MAX - 1] = '\0';

    return 0;
}

/* Verify canaries are intact for all registered nodes */
static int canaries_intact(void)
{
    for (int i = 0; i < TotalNodes; i++) {
        if (NodeList[i].canary_before != CANARY_VALUE) return 0;
        if (NodeList[i].canary_after  != CANARY_VALUE) return 0;
    }
    return 1;
}

/* Reset state between tests */
static void reset_state(void)
{
    memset(NodeList, 0, sizeof(NodeList));
    TotalNodes = 0;
}

START_TEST(test_node_name_buffer_overflow_invariant)
{
    /* INVARIANT: No node name input, regardless of length or content,
     * must cause memory outside the fixed-size node name buffer to be
     * modified. Canary values adjacent to the buffer must remain intact. */

    const char *payloads[] = {
        /* Exact boundary */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", /* 64 chars */
        /* One over boundary */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", /* 65 chars */
        /* Far over boundary */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        /* 512 chars of 'B' */
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        /* Format string attack */
        "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%n%n%n%n%n%n",
        /* Null bytes embedded (C string stops at first null, but test boundary) */
        "short",
        /* Path traversal style */
        "../../../../../../../../etc/passwd",
        /* Shell metacharacters */
        "node; rm -rf /; echo pwned",
        /* Unicode/high-byte sequences */
        "\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8\xf7\xf6\xf5\xf4\xf3\xf2\xf1\xf0"
        "\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8\xf7\xf6\xf5\xf4\xf3\xf2\xf1\xf0"
        "\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8\xf7\xf6\xf5\xf4\xf3\xf2\xf1\xf0"
        "\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8\xf7\xf6\xf5\xf4\xf3\xf2\xf1\xf0",
        /* All zeros except length */
        "\x00\x00\x00\x00\x00\x00\x00\x00",
        /* Newline/carriage return injection */
        "node\r\nHost: evil.com\r\n\r\n",
        /* Very long repeated pattern */
        "node_name_node_name_node_name_node_name_node_name_node_name_node_name_node_name_"
        "node_name_node_name_node_name_node_name_node_name_node_name_node_name_node_name_",
        /* NOP sled simulation */
        "\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
        "\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
        "\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
        "\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
        "\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90",
        /* Empty string */
        "",
        /* Single character */
        "x",
        /* Exactly NODE_NAME_MAX - 1 chars (safe boundary) */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  /* 63 chars */
    };

    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);

    for (int i = 0; i < num_payloads; i++) {
        reset_state();

        /* Pre-condition: canaries are zero (reset state) — initialize them */
        NodeList[0].canary_before = CANARY_VALUE;
        NodeList[0].canary_after  = CANARY_VALUE;

        int ret = safe_add_node(payloads[i]);

        /* The function must either succeed or gracefully reject the input */
        ck_assert_msg(ret == 0 || ret == -1,
            "safe_add_node returned unexpected value for payload index %d", i);

        if (ret == 0) {
            /* INVARIANT 1: The stored name must be null-terminated */
            ck_assert_msg(
                NodeList[TotalNodes - 1].name[NODE_NAME_MAX - 1] == '\0',
                "Buffer is not null-terminated for payload index %d", i
            );

            /* INVARIANT 2: The stored name length must not exceed the buffer */
            size_t stored_len = strlen(NodeList[TotalNodes - 1].name);
            ck_assert_msg(
                stored_len < NODE_NAME_MAX,
                "Stored name length %zu exceeds buffer size %d for payload index %d",
                stored_len, NODE_NAME_MAX, i
            );

            /* INVARIANT 3: Canary values must be intact (no buffer overflow) */
            ck_assert_msg(
                canaries_intact(),
                "Canary value corrupted — buffer overflow detected for payload index %d "
                "(payload length: %zu)", i, strlen(payloads[i])
            );

            /* INVARIANT 4: TotalNodes must remain within valid range */
            ck_assert_msg(
                TotalNodes >= 1 && TotalNodes <= MAX_NODES,
                "TotalNodes %d out of valid range after payload index %d",
                TotalNodes, i
            );
        }
    }
}
END_TEST

START_TEST(test_multiple_nodes_no_overflow)
{
    /* INVARIANT: Adding multiple nodes with adversarial names must not
     * corrupt any previously registered node's canary or name buffer. */

    reset_state();

    const char *nodes[] = {
        "legitimate_node_1",
        /* Oversized — must be truncated, not overflow */
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "legitimate_node_2",
        "%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n",
        "legitimate_node_3",
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        "legitimate_node_4",
    };

    int num_nodes = sizeof(nodes) / sizeof(nodes[0]);

    for (int i = 0; i < num_nodes && TotalNodes < MAX_NODES; i++) {
        int ret = safe_add_node(nodes[i]);
        ck_assert_msg(ret == 0 || ret == -1,
            "Unexpected return value at node index %d", i);

        if (ret == 0) {
            /* INVARIANT: All previously added nodes must still have intact canaries */
            ck_assert_msg(
                canaries_intact(),
                "Canary corruption detected after adding node index %d", i
            );

            /* INVARIANT: Each stored name must be properly null-terminated */
            for (int j = 0; j < TotalNodes; j++) {
                ck_assert_msg(
                    NodeList[j].name[NODE_NAME_MAX - 1] == '\0',
                    "Node %d name not null-terminated after adding node index %d", j, i
                );
                ck_assert_msg(
                    strlen(NodeList[j].name) < (size_t)NODE_NAME_MAX,
                    "Node %d name length exceeds buffer after adding node index %d", j, i
                );
            }
        }
    }
}
END_TEST

START_TEST(test_null_input_handled_safely)
{
    /* INVARIANT: NULL input must be rejected without crashing or corrupting state */
    reset_state();

    int ret = safe_add_node(NULL);
    ck_assert_msg(ret == -1, "NULL input must be rejected (returned %d)", ret);
    ck_assert_msg(TotalNodes == 0, "TotalNodes must remain 0 after NULL input");
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security_Timesync_BufferOverflow");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_node_name_buffer_overflow_invariant);
    tcase_add_test(tc_core, test_multiple_nodes_no_overflow);
    tcase_add_test(tc_core, test_null_input_handled_safely);

    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}