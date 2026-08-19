/**
 * Host used for LLM HTTP (probe, Showcase, DecodeBench, connectivity test).
 *
 * Local Sparks probe loopback: engines like ds4-server (Entrpi/ds4-on-spark
 * via ~/models/ds4f/start.sh) default to `--host 127.0.0.1`, so probing the
 * LAN IP would miss them. Remote Sparks still use lanIp (they must bind a
 * reachable interface or sit behind a tunnel).
 *
 * When running in Docker with port mapping (not network_mode: host), the
 * container cannot reach the host's 127.0.0.1. In that case HOST_ROOT_PATH
 * is set and we fall back to lanIp for local sparks.
 *
 * @param {{ isLocal?: boolean, lanIp?: string } | null | undefined} spark
 * @returns {string}
 */
export function llmProbeHost(spark) {
  if (spark?.isLocal) {
    // In Docker with port mapping, 127.0.0.1 is the container's loopback.
    // HOST_ROOT_PATH indicates a containerized deployment; use lanIp instead.
    if (process.env.HOST_ROOT_PATH) {
      const ip = spark?.lanIp != null ? String(spark.lanIp).trim() : "";
      if (ip) return ip;
    }
    return "127.0.0.1";
  }
  const ip = spark?.lanIp != null ? String(spark.lanIp).trim() : "";
  return ip;
}
